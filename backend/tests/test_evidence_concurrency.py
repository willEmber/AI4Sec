"""Concurrent evidence writes: overlapping batches must queue, never deadlock.

Two searches issued in one step can return the same passage, so their batches
hold the same evidence ids. Each `INSERT … ON CONFLICT DO NOTHING` row locks its
key until the batch commits; batches taking the same keys in different orders
deadlocked (seen when an eval ran six turns at once).
"""

from __future__ import annotations

import asyncio
import random
import unittest

from app.models.evidence_models import SourceLevel


class OverlappingBatchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from tests.pg_support import open_fresh_database

        await open_fresh_database(self)

    async def test_batches_sharing_ids_in_any_order_all_land(self) -> None:
        from app.db import agent_repository as repo
        from app.services.evidence_service import _build

        evidences = [
            _build(source_level=SourceLevel.METADATA, quote=f"Fact {i}", owner_id="o", session_id="s")
            for i in range(60)
        ]
        rng = random.Random(3)
        for _ in range(5):
            batches = [rng.sample(evidences, len(evidences)) for _ in range(4)]
            results = await asyncio.gather(
                *(repo.insert_evidence_many(b) for b in batches), return_exceptions=True
            )
            self.assertEqual([r for r in results if r is not None], [])

        stored = await repo.get_evidence_many([e.evidence_id for e in evidences])
        self.assertEqual(len(stored), len(evidences))

    async def test_a_batch_may_repeat_an_id(self) -> None:
        from app.db import agent_repository as repo
        from app.services.evidence_service import _build

        one = _build(source_level=SourceLevel.METADATA, quote="Fact", owner_id="o", session_id="s")
        await repo.insert_evidence_many([one, one])
        self.assertEqual(len(await repo.get_evidence_many([one.evidence_id])), 1)


if __name__ == "__main__":
    unittest.main()
