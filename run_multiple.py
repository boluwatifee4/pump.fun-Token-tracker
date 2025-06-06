#!/usr/bin/env python3
"""
🔄 Multiple Run Orchestrator
Runs pf_tf.py and pf_tracker.py in sequence multiple times
"""

import asyncio
import subprocess
import sys
import time
from datetime import datetime
import logging
import os


class MultipleRunOrchestrator:
    def __init__(self, max_runs=20):
        self.max_runs = max_runs
        self.current_run = 0
        self.log = self._init_logger()
        self.active_trackers = []

    def _init_logger(self):
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(
                    f"logs/orchestrator_{time.strftime('%Y%m%d')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger("orchestrator")

    async def run_fetch_and_track(self):
        """Run one iteration of fetch and track"""
        try:
            self.current_run += 1
            self.log.info(f"🔄 Starting run {self.current_run}/{self.max_runs}")

            # Run token fetcher
            self.log.info("📡 Running token fetcher...")
            fetcher = subprocess.Popen([sys.executable, "pf_tf.py"])
            await asyncio.sleep(5)  # Give it time to start

            # Wait for fetcher to complete
            fetcher.wait()
            if fetcher.returncode != 0:
                self.log.error("❌ Token fetcher failed!")
                return

            # Start tracker in a new process
            self.log.info("🚀 Starting tracker...")
            tracker = subprocess.Popen([sys.executable, "pf2.py"])
            self.active_trackers.append(tracker)

            # Don't wait for tracker to complete - it will run in background
            self.log.info(f"✅ Run {self.current_run} initiated successfully")

        except Exception as e:
            self.log.error(f"Error in run {self.current_run}: {e}")

    async def orchestrate(self):
        """Main orchestration loop"""
        try:
            while self.current_run < self.max_runs:
                await self.run_fetch_and_track()

                # Wait before starting next run
                await asyncio.sleep(60)  # 1 minute between runs

                # Clean up completed trackers
                self.active_trackers = [
                    t for t in self.active_trackers if t.poll() is None]
                self.log.info(
                    f"📊 Active trackers: {len(self.active_trackers)}")

        except KeyboardInterrupt:
            self.log.info("\n⏹ Orchestrator stopped by user")
        finally:
            # Log final status but don't kill running trackers
            self.log.info(f"📈 Completed {self.current_run} runs")
            self.log.info(
                f"🔄 {len(self.active_trackers)} trackers still running")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run multiple fetch and track cycles")
    parser.add_argument("--runs", type=int, default=20,
                        help="Number of runs to perform")
    args = parser.parse_args()

    orchestrator = MultipleRunOrchestrator(max_runs=args.runs)
    asyncio.run(orchestrator.orchestrate())
