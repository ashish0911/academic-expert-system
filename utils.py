import re
import time
import arxiv
import requests
import streamlit as st
from io import BytesIO
from pypdf import PdfReader
from engine import load_paper_registry

def fetch_and_process_papers(queries):
    client = arxiv.Client()
    all_data = []
    registry = load_paper_registry() # Load processed papers tracker
    
    progress_bar = st.progress(0, text="Checking cache and querying arXiv...")
    total_expected = len(queries) * 2
    count = 0

    for q in queries:
        time.sleep(3) # Politeness delay for arXiv limits
        search = arxiv.Search(query=q.strip(), max_results=2, sort_by=arxiv.SortCriterion.Relevance)
        
        try:
            results = list(client.results(search))
            for result in results:
                count += 1
                paper_id = result.get_short_id()
                
                # --- CACHE DE-DUPLICATION CHECK ---
                if paper_id in registry:
                    progress_bar.progress(min(count / total_expected, 1.0), text=f"Skipping cached: {result.title[:20]}...")
                    continue
                
                progress_bar.progress(min(count / total_expected, 1.0), text=f"Downloading new: {result.title[:20]}...")
                
                resp = requests.get(result.pdf_url)
                reader = PdfReader(BytesIO(resp.content))
                text = "\n".join([page.extract_text() for page in reader.pages])
                
                all_data.append({
                    "title": result.title,
                    "id": paper_id,
                    "url": result.pdf_url,
                    "text": text
                })
        except arxiv.HTTPError as e:
            if "429" in str(e):
                time.sleep(10)
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