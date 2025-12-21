"""
Utility script to manually build and manage embeddings index.
Run this before starting the Streamlit app.
"""

import sys
from embeddings.build_index import build_and_save_index


def main():
    """Main entry point for index building."""
    print("🔍 VectorSeek Index Builder")
    print("===========================")
    print()

    documents_dir = "data/documents"
    output_dir = "indexes"

    print(f"📂 Documents directory: {documents_dir}")
    print(f"💾 Output directory: {output_dir}")
    print()

    try:
        build_and_save_index(documents_dir, output_dir)
        print()
        print("✅ Index built successfully!")
        print()
        print("🚀 You can now run: streamlit run app.py")

    except Exception as e:
        print(f"❌ Error building index: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
