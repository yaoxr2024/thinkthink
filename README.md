#### Client for accessing ThinkThinkSyn AI API Service.

Example:
```python
import asyncio

from thinkthinksyn import ThinkThinkSyn
tts = ThinkThinkSyn(apikey=os.getenv('TTS_APIKEY', ''))

async def test():
    return (await tts.completion(prompt='1+1? tell me ans directly without words.'))['text'].strip()

if __name__ == "__main__":
    asyncio.run(test())
```
