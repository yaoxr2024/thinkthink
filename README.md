## `history_rag` 目录中的改动
在 **`thinkthinksyn/history_rag`** 目录中进行了以下改动

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
