"""
Test OpenRouter rate limits by gradually increasing concurrency.

Usage:
    python test_rate_limit.py --start 10 --max 100 --increment 10
"""

import asyncio
import argparse
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
import time

load_dotenv()

MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek/deepseek-r1")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)


async def make_request():
    """Make a simple test request."""
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=10,
                temperature=0.0,
                messages=[{"role": "user", "content": "Say 'test' and nothing else."}]
            ),
            timeout=30.0
        )
        return True, None
    except Exception as e:
        return False, str(e)


async def test_concurrency(n):
    """Test with N concurrent requests."""
    print(f"\nTesting {n} concurrent requests...")

    start_time = time.time()
    tasks = [make_request() for _ in range(n)]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    successes = sum(1 for success, _ in results if success)
    failures = n - successes
    errors = [err for success, err in results if not success]

    print(f"  ✓ {successes}/{n} succeeded in {elapsed:.2f}s")
    print(f"  ✗ {failures}/{n} failed")

    if errors:
        print(f"  Errors: {errors[:3]}")  # Show first 3 errors

    # Check for rate limit errors
    rate_limit_errors = [e for e in errors if "429" in e or "rate" in e.lower()]
    if rate_limit_errors:
        print(f"  ⚠ Hit rate limit!")
        return False

    return successes == n


async def main():
    parser = argparse.ArgumentParser(description='Test OpenRouter rate limits')
    parser.add_argument('--start', type=int, default=10)
    parser.add_argument('--max', type=int, default=100)
    parser.add_argument('--increment', type=int, default=10)
    args = parser.parse_args()

    print(f"Testing OpenRouter rate limits for {MODEL_NAME}")
    print(f"Starting at {args.start}, incrementing by {args.increment}, max {args.max}\n")

    for n in range(args.start, args.max + 1, args.increment):
        success = await test_concurrency(n)

        if not success:
            print(f"\n✗ Rate limit hit at {n} concurrent requests")
            print(f"  Safe limit appears to be ~{n - args.increment} concurrent requests")
            break

        # Small delay between tests
        await asyncio.sleep(2)
    else:
        print(f"\n✓ Successfully tested up to {args.max} concurrent requests!")
        print(f"  You can probably go higher. Try --max {args.max * 2}")


if __name__ == "__main__":
    asyncio.run(main())
