
agent_image_extractor = Agent(
    llm = llm_sverd,   #getting some error message when defining llm, both for groq and gemini
    role="Image Text Extraction Specialist",
    goal="Thoroughly analyze any pictures you receive",
    backstory=(
        """You are an expert in text extraction, specializing in using AI to process and analyze textual content from images. 
            Make sure you use the tools provided."""
    ),
    # tools=[search_tool, scrape_tool, pdf_search_tool, read_directory_tool],
    verbose=True,
    memory=True,
)