from itertools import batched
from json import dumps, loads
from os import getenv
from sys import argv

import stamina
from aiolimiter import AsyncLimiter
from anyio import Path
from asyncer import create_task_group, runnify
from httpx2 import AsyncClient, HTTPError
from loguru import logger
from rich.progress import track
from stamina.instrumentation import set_on_retry_hooks


async def main():
    _, questions_path, *items_paths = argv

    if not await Path(questions_path).exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    for item_path in items_paths:
        if not await Path(item_path).exists():
            raise FileNotFoundError(f"Sentences file not found: {item_path}")

    client = AsyncClient(
        base_url="https://openrouter.ai",
        headers={"Authorization": f"Bearer {getenv('OPENROUTER_API_KEY')}"},
        http2=True,
    )

    questions = loads(await Path(questions_path).read_bytes())

    limiter = AsyncLimiter(1000)

    async def decide(path: Path, state: str):
        @stamina.retry(on=HTTPError)
        async def request():
            response = await client.post(
                "/api/alpha/decisions",
                json={
                    "model": "typesafe/jev-1.13",
                    "state": state,
                    "questions": questions,
                },
            )
            response.raise_for_status()
            return response.json()

        if await path.exists():
            logger.warning(f"Decision already exists for path: {path}")
            return

        await path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with limiter:
                decision = await request()
            await path.write_text(
                dumps(
                    {"state": state, "decision": decision},
                    ensure_ascii=False,
                )
            )
        except Exception:  # noqa: BLE001
            logger.exception(f"Failed to decide for path: {path}")
            return

    for item_path in items_paths:
        batches = list(batched(loads(await Path(item_path).read_bytes()), 1000))
        for batch in track(batches):
            async with create_task_group() as task_group:
                for item in batch:
                    for index, sentence in enumerate(item["sentences"]):
                        path = Path(f"data/decisions/{item['id']}/{index}.json")
                        task_group.soonify(decide)(path, sentence)

    await client.aclose()


logger.add(
    "data/logs/decide.jsonl",
    serialize=True,
    rotation="1 GB",
    compression="gz",
)
logger.add(
    "data/logs/decide-error.jsonl",
    level="ERROR",
    serialize=True,
    rotation="1 GB",
    compression="gz",
)
set_on_retry_hooks([lambda details: logger.warning(details)])
runnify(main)()
