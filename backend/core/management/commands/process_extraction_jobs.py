import time
from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.db import connection

from core.services.extraction_jobs import DEFAULT_LEASE_TIMEOUT, run_one_job


class Command(BaseCommand):
    help = (
        "Process durable founder-message extraction jobs. Requires PostgreSQL; "
        "knowledge publication remains blocked until the sensitive-content guard is integrated."
    )

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true", help="Process at most one eligible job and exit.")
        parser.add_argument("--poll-interval", type=float, default=5.0,
                            help="Seconds to wait when no job is eligible (default: 5).")
        parser.add_argument("--lease-timeout", type=int, default=int(DEFAULT_LEASE_TIMEOUT.total_seconds()),
                            help="Seconds before a processing job may be recovered as stale (default: 600).")

    def handle(self, *args, **options):
        if connection.vendor != "postgresql":
            raise CommandError("process_extraction_jobs requires PostgreSQL row locking.")
        if options["poll_interval"] <= 0 or options["lease_timeout"] <= 0:
            raise CommandError("poll interval and lease timeout must be positive.")

        lease_timeout = timedelta(seconds=options["lease_timeout"])
        while True:
            result = run_one_job(lease_timeout=lease_timeout)
            if result is not None:
                self.stdout.write(f"job={result.job_id} status={result.status} error_code={result.error_code}")
            elif options["once"]:
                self.stdout.write("No eligible extraction jobs.")
            if options["once"]:
                return
            if result is None:
                time.sleep(options["poll_interval"])
