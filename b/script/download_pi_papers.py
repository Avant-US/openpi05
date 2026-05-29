#!/usr/bin/env python3
"""
Scan pi_docls_2.md for papers/articles in sections 一、二、三,
extract URLs, and download PDFs + web pages to p/<sanitized_title>/.
"""

import hashlib
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MD_FILE = BASE_DIR / "b" / "d" / "pi_docls_2.md"
OUTPUT_DIR = BASE_DIR / "p"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 5


def sanitize_folder_name(title: str) -> str:
    title = title.strip()
    title = re.sub(r'[<>:"/\\|?*]', '', title)
    title = title.replace(' ', '_')
    title = re.sub(r'_+', '_', title)
    title = title.strip('_.')
    if len(title) > 200:
        title = title[:200]
    return title


def parse_sections(md_text: str) -> list[dict]:
    target_sections = [
        "## 一、Physical Intelligence 正式发表的论文 & 技术报告（由新到旧，完整版）",
        "## 二、PI 成员在公司成立前后的关键前置/关联研究（由新到旧）",
        "## 三、博客文章 & 公告（由新到旧，完整版）",
    ]

    lines = md_text.split('\n')
    section_ranges = []

    for target in target_sections:
        start_idx = None
        for i, line in enumerate(lines):
            if line.strip() == target:
                start_idx = i
                break
        if start_idx is None:
            print(f"WARNING: Section not found: {target}")
            continue

        end_idx = len(lines)
        for i in range(start_idx + 1, len(lines)):
            if lines[i].startswith('## ') and not lines[i].startswith('### '):
                end_idx = i
                break
            if lines[i].startswith('---'):
                check_next = False
                for j in range(i + 1, min(i + 3, len(lines))):
                    if lines[j].startswith('## ') and not lines[j].startswith('### '):
                        check_next = True
                        break
                if check_next:
                    end_idx = i
                    break

        section_ranges.append((start_idx, end_idx))

    entries = []
    for sec_start, sec_end in section_ranges:
        subsection_starts = []
        for i in range(sec_start, sec_end):
            if lines[i].startswith('### '):
                subsection_starts.append(i)

        for idx, ss_start in enumerate(subsection_starts):
            if idx + 1 < len(subsection_starts):
                ss_end = subsection_starts[idx + 1]
            else:
                ss_end = sec_end

            title_line = lines[ss_start]
            title = re.sub(r'^###\s*\d+\.\s*', '', title_line).strip()

            block = '\n'.join(lines[ss_start:ss_end])

            urls = re.findall(r'<(https?://[^>]+)>', block)
            url_set = []
            seen = set()
            for u in urls:
                if u not in seen:
                    seen.add(u)
                    url_set.append(u)

            entries.append({
                'title': title,
                'folder': sanitize_folder_name(title),
                'urls': url_set,
                'block': block,
            })

    return entries


def classify_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    path_lower = parsed.path.lower()
    if path_lower.endswith('.pdf'):
        return 'pdf'
    if 'arxiv.org/abs/' in url:
        return 'arxiv'
    if 'github.com' in parsed.netloc:
        return 'webpage'
    if 'openreview.net' in parsed.netloc:
        return 'webpage'
    if 'proceedings.mlr.press' in parsed.netloc and path_lower.endswith('.pdf'):
        return 'pdf'
    if 'proceedings.mlr.press' in parsed.netloc:
        return 'webpage'
    if 'huggingface.co' in parsed.netloc:
        return 'webpage'
    if 'roboticsproceedings.org' in parsed.netloc and path_lower.endswith('.pdf'):
        return 'pdf'
    return 'webpage'


def get_arxiv_pdf_url(abs_url: str) -> str:
    return abs_url.replace('/abs/', '/pdf/') + '.pdf'


def download_file(url: str, filepath: Path) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True, allow_redirects=True)
            resp.raise_for_status()
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  OK: {filepath.name} ({filepath.stat().st_size / 1024:.1f} KB)")
            return True
        except Exception as e:
            print(f"  RETRY {attempt}/{MAX_RETRIES} for {url}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    print(f"  FAILED: {url}")
    return False


def url_to_filename(url: str, ext: str = '') -> str:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.strip('/')
    if path:
        name = path.replace('/', '_')
    else:
        name = parsed.netloc.replace('.', '_')
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    if ext and not name.lower().endswith(ext.lower()):
        name += ext
    if len(name) > 150:
        h = hashlib.md5(url.encode()).hexdigest()[:8]
        name = name[:140] + '_' + h + ext
    return name


def download_webpage_with_resources(url: str, folder: Path, filename_base: str) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
            resp.raise_for_status()
            break
        except Exception as e:
            print(f"  RETRY {attempt}/{MAX_RETRIES} for {url}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                print(f"  FAILED webpage: {url}")
                return False

    content_type = resp.headers.get('Content-Type', '')
    if 'pdf' in content_type.lower() or resp.content[:5] == b'%PDF-':
        pdf_path = folder / (filename_base + '.pdf')
        if not pdf_path.exists():
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pdf_path, 'wb') as f:
                f.write(resp.content)
            print(f"  OK (redirected PDF): {pdf_path.name} ({pdf_path.stat().st_size / 1024:.1f} KB)")
        return True

    html_content = resp.text
    soup = BeautifulSoup(html_content, 'html.parser')

    html_path = folder / (filename_base + '.html')
    folder.mkdir(parents=True, exist_ok=True)

    res_folder = folder / (filename_base + '_files')
    res_folder.mkdir(parents=True, exist_ok=True)

    resource_tags = []
    for img in soup.find_all('img', src=True):
        resource_tags.append(('src', img))
    for link in soup.find_all('link', href=True):
        rel = link.get('rel', [])
        if isinstance(rel, list):
            rel = ' '.join(rel)
        if 'stylesheet' in rel or 'icon' in rel:
            resource_tags.append(('href', link))
    for script in soup.find_all('script', src=True):
        resource_tags.append(('src', script))
    for video in soup.find_all('video'):
        if video.get('src'):
            resource_tags.append(('src', video))
        for source in video.find_all('source', src=True):
            resource_tags.append(('src', source))

    downloaded_resources = 0
    for attr_name, tag in resource_tags:
        res_url = tag[attr_name]
        if not res_url or res_url.startswith('data:'):
            continue

        abs_res_url = urllib.parse.urljoin(url, res_url)
        res_filename = url_to_filename(abs_res_url)

        res_path = res_folder / res_filename
        if not res_path.exists():
            try:
                r = requests.get(abs_res_url, headers=HEADERS, timeout=30, allow_redirects=True)
                r.raise_for_status()
                with open(res_path, 'wb') as f:
                    f.write(r.content)
                downloaded_resources += 1
            except Exception as e:
                print(f"    Skip resource {abs_res_url}: {e}")
                continue

        rel_path = f"{filename_base}_files/{res_filename}"
        tag[attr_name] = rel_path

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(str(soup))

    if downloaded_resources == 0 and not any(res_folder.iterdir()):
        res_folder.rmdir()

    fsize = html_path.stat().st_size / 1024
    print(f"  OK: {html_path.name} ({fsize:.1f} KB, {downloaded_resources} resources)")
    return True


def process_entry(entry: dict) -> dict:
    title = entry['title']
    folder_name = entry['folder']
    urls = entry['urls']

    folder = OUTPUT_DIR / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    results = {'title': title, 'ok': [], 'failed': []}

    for url in urls:
        url_type = classify_url(url)

        if url_type == 'pdf':
            fname = url_to_filename(url, '.pdf')
            fpath = folder / fname
            if fpath.exists() and fpath.stat().st_size > 1000:
                print(f"  SKIP (exists): {fpath.name}")
                results['ok'].append(url)
                continue
            if download_file(url, fpath):
                results['ok'].append(url)
            else:
                results['failed'].append(url)

        elif url_type == 'arxiv':
            fname_base = url_to_filename(url)
            html_path = folder / (fname_base + '.html')
            if not html_path.exists():
                if download_webpage_with_resources(url, folder, fname_base):
                    results['ok'].append(url)
                else:
                    results['failed'].append(url)
            else:
                print(f"  SKIP (exists): {html_path.name}")
                results['ok'].append(url)

            pdf_url = get_arxiv_pdf_url(url)
            pdf_fname = url_to_filename(pdf_url, '.pdf')
            pdf_path = folder / pdf_fname
            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                print(f"  SKIP (exists): {pdf_path.name}")
                results['ok'].append(pdf_url)
            else:
                print(f"  Downloading arXiv PDF: {pdf_url}")
                if download_file(pdf_url, pdf_path):
                    results['ok'].append(pdf_url)
                else:
                    results['failed'].append(pdf_url)

        elif url_type == 'webpage':
            fname_base = url_to_filename(url)
            html_path = folder / (fname_base + '.html')
            if html_path.exists():
                print(f"  SKIP (exists): {html_path.name}")
                results['ok'].append(url)
                continue
            if download_webpage_with_resources(url, folder, fname_base):
                results['ok'].append(url)
            else:
                results['failed'].append(url)

    return results


def main():
    print(f"Reading: {MD_FILE}")
    md_text = MD_FILE.read_text(encoding='utf-8')

    entries = parse_sections(md_text)
    print(f"Found {len(entries)} entries to process.\n")

    all_failed = []
    for i, entry in enumerate(entries, 1):
        print(f"\n[{i}/{len(entries)}] {entry['title']}")
        print(f"  Folder: p/{entry['folder']}/")
        print(f"  URLs: {len(entry['urls'])}")

        result = process_entry(entry)
        if result['failed']:
            all_failed.append(result)

        time.sleep(1)

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    total_urls = sum(len(e['urls']) for e in entries)
    total_arxiv = sum(1 for e in entries for u in e['urls'] if 'arxiv.org/abs/' in u)
    print(f"Entries: {len(entries)}")
    print(f"Total URLs (excluding auto-generated arXiv PDFs): {total_urls}")
    print(f"ArXiv papers (each generates an extra PDF download): {total_arxiv}")

    if all_failed:
        print(f"\nFAILED DOWNLOADS ({len(all_failed)} entries had failures):")
        for r in all_failed:
            print(f"  {r['title']}:")
            for u in r['failed']:
                print(f"    - {u}")
        sys.exit(1)
    else:
        print("\nAll downloads completed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
