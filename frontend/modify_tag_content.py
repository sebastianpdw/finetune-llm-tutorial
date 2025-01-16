import logging
import pathlib
import shutil
from bs4 import BeautifulSoup
import streamlit as st

# SEO Configuration
SEO_CONFIG = {
    "title": "Lex Llama - Your AI Fitness Motivation Coach",
    "description": "Transform your fitness journey with Lex Llama, your personal AI motivation coach. Stay consistent with your fitness goals and find the drive to push through any challenge. Your 24/7 companion for fitness motivation.",
    "keywords": "AI fitness coach, workout motivation, fitness goals, personal trainer AI, exercise motivation, fitness journey, workout companion, AI motivation coach, fitness transformation, workout consistency",
    "author": "Sebastian Panman de Wit",
    "og_image": "https://media.licdn.com/dms/image/v2/D5622AQG4vCBJm7JKeg/feedshare-shrink_800/B56ZRJoJ5zG8Ak-/0/1736402051334?e=1739404800&v=beta&t=HyH-FmlShTj2VnTgHVOIOIiosUs4sOCEmukbLWRwXBM",
    "twitter_card": "summary_large_image",
}


def modify_meta_tags():
    """Modify the Streamlit index.html to include SEO meta tags"""
    index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
    logging.info(f"editing {index_path}")

    # Read the original file
    soup = BeautifulSoup(index_path.read_text(), features="html.parser")

    # Backup the original file if backup doesn't exist
    bck_index = index_path.with_suffix(".bck")
    if not bck_index.exists():
        shutil.copy(index_path, bck_index)

    # Remove existing meta tags we're going to replace
    for tag in soup.find_all(["title", "meta"]):
        if tag.get("name") in ["description", "keywords", "author"] or tag.get(
            "property", ""
        ).startswith(("og:", "twitter:")):
            tag.decompose()

    # Remove existing favicon
    for tag in soup.find_all("link", rel="icon"):
        tag.decompose()

    # Helper function to create and append tags
    def add_or_update_tag(tag_type, attrs=None, content=None):
        new_tag = soup.new_tag(tag_type)
        if attrs:
            for key, value in attrs.items():
                new_tag[key] = value
        if content:
            new_tag.string = content
        soup.head.append(new_tag)

    # Add title
    add_or_update_tag("title", content=SEO_CONFIG["title"])

    # Add favicon
    add_or_update_tag(
        "link",
        {
            "rel": "icon",
            "type": "image/x-icon",
            "href": "https://media.licdn.com/dms/image/v2/D5622AQG4vCBJm7JKeg/feedshare-shrink_800/B56ZRJoJ5zG8Ak-/0/1736402051334?e=1739404800&v=beta&t=HyH-FmlShTj2VnTgHVOIOIiosUs4sOCEmukbLWRwXBM",
        },
    )

    # Add basic meta tags
    add_or_update_tag(
        "meta", {"name": "description", "content": SEO_CONFIG["description"]}
    )
    add_or_update_tag("meta", {"name": "keywords", "content": SEO_CONFIG["keywords"]})
    add_or_update_tag("meta", {"name": "author", "content": SEO_CONFIG["author"]})

    # Add Open Graph tags
    og_tags = {
        "og:type": "website",
        "og:title": SEO_CONFIG["title"],
        "og:description": SEO_CONFIG["description"],
        "og:image": SEO_CONFIG["og_image"],
    }
    for prop, content in og_tags.items():
        add_or_update_tag("meta", {"property": prop, "content": content})

    # Add Twitter Card tags
    twitter_tags = {
        "twitter:card": SEO_CONFIG["twitter_card"],
        "twitter:title": SEO_CONFIG["title"],
        "twitter:description": SEO_CONFIG["description"],
        "twitter:image": SEO_CONFIG["og_image"],
    }
    for name, content in twitter_tags.items():
        add_or_update_tag("meta", {"name": name, "content": content})

    # Save the modified HTML
    index_path.write_text(str(soup))
    logging.info("SEO tags successfully updated")


# Example usage
if __name__ == "__main__":
    modify_meta_tags()
