import os
import sys
import logging
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
import openai
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from langdetect import detect
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import re

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.error("OpenAI API key not found!")
    sys.exit(1)

# Your Telegram Bot Token
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    logger.error("Telegram token not found!")
    sys.exit(1)

# Your Telegram Chat ID
MY_CHAT_ID = os.getenv("MY_CHAT_ID")
if not MY_CHAT_ID:
    logger.error("Chat ID not found!")
    sys.exit(1)

# Configure requests to retry on failure
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("http://", adapter)
http.mount("https://", adapter)

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu.be\/|youtube.com\/shorts\/)([^&\n?]*)',
        r'(?:youtube\.com\/embed\/)([^&\n?]*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(video_id: str) -> tuple:
    """Get transcript in English or Italian, returns (transcript_text, language)."""
    try:
        # First try to get all available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get English transcript first
        try:
            transcript = transcript_list.find_transcript(['en'])
            return TextFormatter().format_transcript(transcript.fetch()), 'en'
        except:
            # Try to get Italian transcript
            try:
                transcript = transcript_list.find_transcript(['it'])
                return TextFormatter().format_transcript(transcript.fetch()), 'it'
            except:
                # If neither English nor Italian is available, try to get any transcript and translate to English
                try:
                    transcript = transcript_list.find_transcript(['en', 'it'])
                    transcript = transcript.translate('en')
                    return TextFormatter().format_transcript(transcript.fetch()), 'en'
                except Exception as e:
                    logger.error(f"Error getting transcript: {str(e)}")
                    return None, None
    except Exception as e:
        logger.error(f"Error accessing transcript: {str(e)}")
        return None, None

def summarize_youtube_video(url: str) -> str:
    try:
        logger.info(f"Processing YouTube URL: {url}")
        video_id = extract_video_id(url)
        if not video_id:
            return "Error: Could not extract YouTube video ID from the URL."
        
        transcript, lang = get_youtube_transcript(video_id)
        if not transcript:
            return "Error: Could not retrieve transcript for this video."
        
        system_msg = "You are an assistant that creates structured video summaries."
        user_prompt = f"Analyze the following transcript and create a structured summary:\n{transcript}"

        logger.info("Sending request to OpenAI API for video summary")
        JSON_DATA = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 1000  # Reduced token limit
        }

        api_response = http.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai.api_key}", "Content-Type": "application/json"},
            json=JSON_DATA,
            timeout=30
        )
        
        api_response.raise_for_status()
        summary = api_response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        
        if not summary:
            logger.warning("No summary received from OpenAI API")
            return "Error: No summary received from the API."
        
        video_title = "YouTube Video"
        logger.info("Successfully generated video summary")
        return f"📺 _{video_title}_\n\n{summary}"
    
    except Exception as e:
        logger.error(f"Error in summarize_youtube_video: {str(e)}", exc_info=True)
        return f"An error occurred while processing the YouTube video: {str(e)}"

def is_youtube_url(url: str) -> bool:
    """Check if the URL is a YouTube video URL."""
    youtube_patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu.be\/|youtube.com\/shorts\/)',
        r'(?:youtube\.com\/embed\/)',
    ]
    return any(re.search(pattern, url) for pattern in youtube_patterns)

def get_language_config(text: str) -> tuple:
    """
    Detect language and return appropriate system message and user prompt.
    """
    try:
        lang = detect(text)
        logger.info(f"Detected language: {lang}")
    except Exception as e:
        logger.warning(f"Language detection failed: {str(e)}. Defaulting to English.")
        lang = 'en'
    
    if lang == 'it':
        system_msg = "Sei un assistente che fornisce riassunti dettagliati in italiano."
        user_prompt = "Fornisci un riassunto dettagliato in italiano del seguente testo:\n\n{}\n\nIl riassunto deve essere completo e dettagliato, mantenendo lo stesso tono dell'articolo originale."
    else:
        system_msg = "You are a helpful assistant that provides detailed summaries in English."
        user_prompt = "Provide a detailed summary of the following text:\n\n{}\n\nThe summary should be complete and detailed, maintaining the same tone as the original article."
    
    return system_msg, user_prompt

def clean_content(raw_content: str) -> str:
    try:
        cleaned_content = "\n".join(
            line for line in raw_content.splitlines()
            if not line.startswith(">") and "Leggi anche:" not in line
        )
        cleaned_content = cleaned_content.replace("![alt text](url)", "")
        return cleaned_content.strip()
    except Exception as e:
        logger.error(f"Error cleaning content: {str(e)}", exc_info=True)
        return ""

def summarize_url(url: str) -> str:
    try:
        logger.info(f"Fetching content from URL: {url}")
        response = http.get(f"http://mercury:3000/parser?url={url}", timeout=30)
        response.raise_for_status()
        response_json = response.json()

        raw_content = response_json.get('content', '')
        article_title = response_json.get('title', '')

        if not raw_content:
            logger.warning("No content found in the article")
            return "Error: No content found in the article."

        cleaned_content = clean_content(raw_content)

        if not cleaned_content:
            logger.warning("No valid content after cleaning")
            return "Error: No valid content to summarize after cleaning."

        system_msg, user_prompt = get_language_config(cleaned_content)
        
        logger.info("Sending request to OpenAI API")
        JSON_DATA = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt.format(cleaned_content)}
            ],
            "temperature": 0.5,
            "max_tokens": 1000  # Reduced token limit
        }

        api_response = http.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai.api_key}", "Content-Type": "application/json"},
            json=JSON_DATA,
            timeout=30
        )
        
        api_response.raise_for_status()
        summary = api_response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        
        if not summary:
            logger.warning("No summary received from OpenAI API")
            return "Error: No summary received from the API."

        logger.info("Successfully generated summary")
        return f"📰 _{article_title}_\n\n{summary}"
    
    except requests.Timeout:
        logger.error("Request timed out")
        return "Error: Request timed out. Please try again."
    except requests.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return f"Error occurred while processing the URL: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return f"An unexpected error occurred: {str(e)}"

def grammar_proof_text(text: str) -> tuple:
    try:
        lang = detect(text)
        logger.info(f"Detected language for grammar check: {lang}")
    except:
        logger.warning("Language detection failed for grammar check. Defaulting to English.")
        lang = 'en'

    if lang == 'it':
        system_msg = "Sei un assistente che corregge la grammatica mantenendo il tono e lo stile originale del testo."
        user_prompt = f"Correggi il seguente testo per grammatica e chiarezza e illustra le modifiche apportate dopo un segnaposto %changes%, mentre non inserire nulla nel test corretto:\\n\\n{text}"
    else:
        system_msg = "You are an assistant that corrects grammar while maintaining the original tone and style."
        user_prompt = f"Proofread the following text for grammar and clarity and illustrate changes made after using this placemark %changes%, while not including anything in the corrected text:\\n\\n{text}"

    logger.info("Sending request to OpenAI API for grammar correction")
    JSON_DATA = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 750
    }

    correction_response = http.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {openai.api_key}", "Content-Type": "application/json"},
        json=JSON_DATA,
        timeout=30
    )
    
    correction_response.raise_for_status()
    corrected_text = correction_response.json().get('choices', [{}])[0].get('message', {}).get('content', '')

    if not corrected_text:
        logger.warning("No correction received from OpenAI API")
        return "Error: No correction received from the API.", ""

    changes_start = corrected_text.find("%changes%")
    changes_explanation = corrected_text[changes_start + len("%changes%"):] if changes_start != -1 else ""
    corrected_text = corrected_text[:changes_start].strip() if changes_start != -1 else corrected_text.strip()

    logger.info("Successfully generated grammar correction")
    return corrected_text, changes_explanation

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        if str(update.effective_chat.id) == MY_CHAT_ID:
            await update.message.reply_text('Send me a URL for summarization or generic text for grammar correction!')
        else:
            logger.warning(f"Unauthorized access attempt from chat ID: {update.effective_chat.id}")
            await update.message.reply_text("I'm sorry, but I can only talk to the bot owner.")
    except Exception as e:
        logger.error(f"Error in start command: {str(e)}", exc_info=True)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        if str(update.effective_chat.id) == MY_CHAT_ID:
            message_text = update.message.text
            logger.info(f"Received message: {message_text[:100]}...")  # Log first 100 chars

            if message_text.startswith("http://") or message_text.startswith("https://"):
                if is_youtube_url(message_text):
                    summary = summarize_youtube_video(message_text)
                else:
                    summary = summarize_url(message_text)
                
                # Split long messages
                message_parts = split_long_message(summary)
                for part in message_parts:
                    await update.message.reply_text(part, parse_mode='Markdown')
            else:
                corrected_text, changes_explanation = grammar_proof_text(message_text)
                # Send the corrected text first
                await update.message.reply_text(corrected_text)
                # Then send the explanation of changes
                if changes_explanation:
                    # Split explanation if it's too long
                    explanation_parts = split_long_message("\n📝 Changes made:\n" + changes_explanation)
                    for part in explanation_parts:
                        await update.message.reply_text(part)
        else:
            logger.warning(f"Unauthorized message from chat ID: {update.effective_chat.id}")
            await update.message.reply_text("I'm sorry, but I can only talk to the bot owner.")
    except Exception as e:
        logger.error(f"Error handling message: {str(e)}", exc_info=True)
        await update.message.reply_text("An error occurred while processing your message. Please try again later.")

def split_long_message(text: str, max_length: int = 4000) -> list:
    """Split a long message into parts that fit within Telegram's message length limit."""
    # If the text starts with a title (indicated by _), preserve it in each part
    title = ""
    content = text
    if text.startswith("📺 _") or text.startswith("📰 _"):
        title_end = text.find("_\n\n")
        if title_end != -1:
            title = text[:title_end+3] + "\n\n"  # Include the "_\n\n"
            content = text[title_end+3:]

    parts = []
    current_part = title
    
    # Split by double newlines to preserve paragraph structure
    paragraphs = content.split('\n\n')
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed the limit
        if len(current_part + paragraph) + 2 > max_length:  # +2 for the \n\n
            if current_part != title:  # Don't append empty parts
                parts.append(current_part.strip())
            current_part = title + paragraph
        else:
            if current_part != title:
                current_part += '\n\n'
            current_part += paragraph
    
    if current_part != title:
        parts.append(current_part.strip())
    
    # Add part numbers if there are multiple parts
    if len(parts) > 1:
        return [f"{parts[i]}\n\n[Part {i+1}/{len(parts)}]" for i in range(len(parts))]
    return parts

def main() -> None:
    try:
        logger.info("Starting bot...")
        application = Application.builder().token(TELEGRAM_TOKEN).build()

        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        logger.info("Bot is ready to handle messages")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)
