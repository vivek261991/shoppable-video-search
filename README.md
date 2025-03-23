# 🛋️ For Ads



###  ✅ Start The App

Milvus is used as a vector database. Run it using the helper script:

```bash
# Download Milvus standalone script
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

# Start Milvus in Docker
bash standalone_embed.sh start

# Build the image
docker build -t shoppable-video .

# Run the app (with Streamlit)
docker run -it --rm -p 8501:8501 shoppable-video

# Transcribe
docker run -it --rm shoppable-video bash
python scripts/transcribe.py --video clip.mp4

# Embed 
docker run -it --rm shoppable-video bash
python scripts/embed.py --video clip.mp4

# Search via container 
docker run -it --rm shoppable-video bash
python scripts/search.py --video clip.mp4 --query "Why did they skip the delivery for the couch?"
