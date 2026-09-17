"""SigV4 request signing for httpx clients that talk to AWS-hosted OpenAI-wire
endpoints (Bedrock's `/openai/v1/…`). The provider SDK builds the request as
usual; this auth hook replaces its bearer header with an AWS signature over
the exact bytes on the wire. Credentials come from the default boto chain
(env / profile / pod role) and are re-resolved per request so rotated role
credentials are picked up without a restart.
"""

from __future__ import annotations

from collections.abc import Generator

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

# Headers SigV4Auth adds; everything else on the httpx request stays as the SDK
# set it (and is not part of SignedHeaders).
_SIGNED = ("Authorization", "X-Amz-Date", "X-Amz-Security-Token")


class SigV4HttpxAuth(httpx.Auth):
    requires_request_body = True

    def __init__(self, region: str, service: str = "bedrock") -> None:
        self._region = region
        self._service = service
        self._session = boto3.Session(region_name=region)

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        creds = self._session.get_credentials()
        if creds is None:
            raise RuntimeError("no AWS credentials available for SigV4 signing")
        aws_req = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content,
            headers={"Content-Type": request.headers.get("content-type", "application/json")},
        )
        SigV4Auth(creds.get_frozen_credentials(), self._service, self._region).add_auth(aws_req)
        signed = aws_req.prepare().headers
        for name in _SIGNED:
            if name in signed:
                request.headers[name] = signed[name]
        yield request
