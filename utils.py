import re
import time
import arxiv
import requests
import streamlit as st
from io import BytesIO
from pypdf import PdfReader

def fetch_and_process_papers(queries):
    client = arxiv.Client()
    all_data = []
    
    progress_bar = st.progress(0, text="Initializing arXiv connection...")
    total_expected = len(queries) * 2
    count = 0

    for q in queries:
        time.sleep(3) 
        
        search = arxiv.Search(
            query=q.strip(), 
            max_results=2, # Keep this low to avoid 429s
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        try:
            # We wrap the results iteration in a try/except for 429s
            results = list(client.results(search))
            
            for result in results:
                count += 1
                progress_bar.progress(min(count / total_expected, 1.0), text=f"Reading: {result.title[:30]}...")
                
                # Download PDF
                resp = requests.get(result.pdf_url)
                reader = PdfReader(BytesIO(resp.content))
                text = "\n".join([page.extract_text() for page in reader.pages])
                
                all_data.append({
                    "title": result.title,
                    "id": result.get_short_id(),
                    "url": result.pdf_url,
                    "text": text
                })
        
        except arxiv.HTTPError as e:
            if "429" in str(e):
                st.error("🚨 Rate limited by arXiv. Waiting 10 seconds before skipping...")
                time.sleep(10)
            else:
                st.error(f"Error fetching from arXiv: {e}")
            continue

    progress_bar.empty()
    return all_data

def get_clickable_markdown(ans):
    text = ans.answer_text
    source_map = {s.citation_number: s for s in ans.sources_used}
    pattern = r'\[(\d+(?:,\s*\d+)*)\]'

    def replace_with_links(match):
        nums = [n.strip() for n in match.group(1).split(',')]
        links = [f"[[{n}]]({source_map[int(n)].url})" for n in nums if n.isdigit() and int(n) in source_map]
        return ", ".join(links) if links else match.group(0)

    final_text = re.sub(pattern, replace_with_links, text)
    ref_section = "\n\n---\n### References\n"
    for num in sorted(source_map.keys()):
        s = source_map[num]
        ref_section += f"{num}. [{s.title}]({s.url}) (arXiv:{s.arxiv_id})\n"
    return final_text + ref_section