# telegram-openai-summarize-proofcheck-bot
A self-hosted Telegram bot that can summarize articles and YouTube videos or proofread text. If it receives a URL with an article or a YouTube video, it generates a summary. If it receives general text, it replies with a proofread version of the same text, also identifying the changes made.

# How to deploy
Download the files from the repo ("git clone https://github.com/mariomaz87/telegram-openai-summarize-proofcheck-bot"), modify docker compose with your details, and deploy with:

docker-compose build

docker-compose up -d

The docker needs to access an active instance of Mercury Parser to extract content for provided urls.
