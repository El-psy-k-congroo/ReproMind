"""Retrieve author-linked repository documentation with independent transports."""
import base64
import binascii
import httpx


async def linked_readme(client, owner, repository):
    errors=[]
    api=f'https://api.github.com/repos/{owner}/{repository}/readme'
    try:
        response=await client.get(api)
        response.raise_for_status()
        payload=response.json()
        content=payload.get('content','')
        if payload.get('encoding')!='base64' or not isinstance(content,str) or len(content)>2_000_000:
            raise ValueError('README 编码或大小不受支持')
        return base64.b64decode(''.join(content.split()),validate=True).decode('utf-8',errors='replace')[:60_000], errors
    except (httpx.HTTPError,ValueError,TypeError,AttributeError,binascii.Error) as exc:
        errors.append(f'{api}: {type(exc).__name__}: {str(exc)[:200]}')
    for branch in ('master','main'):
        url=f'https://raw.githubusercontent.com/{owner}/{repository}/{branch}/README.md'
        try:
            response=await client.get(url)
            response.raise_for_status()
            return response.text[:60_000], errors
        except httpx.HTTPError as exc:
            errors.append(f'{url}: {type(exc).__name__}: {str(exc)[:200]}')
    return '',errors
