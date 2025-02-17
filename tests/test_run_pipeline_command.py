import subprocess
import pytest

def test_run_pipeline():
    """Test that the 'run_pipeline' command runs successfully with a given URL."""
    test_url = "https://api.creativeforagingtask.com/v1/event.csv?game=c9d8979c-94ad-498f-8d2b-a37cff3c5b41&gameVersion=40f2894d-1891-456b-af26-a386c6111287&entityType=event&before=2024-12-05T0:00:00.000Z"

    result = subprocess.run(
        ["run_pipeline", "--url", test_url], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )

    assert result.returncode == 0, f"Command failed: {result.stderr}"

    print(result.stdout)
