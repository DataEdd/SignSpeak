"""Search and filter functionality for signs."""

from typing import Optional

from .models import Sign, SignStatus


class SignSearch:
    """Search and filter signs in the database."""

    def __init__(self, store: "SignStore"):
        """Initialize with a SignStore instance."""
        self.store = store

    def search(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        min_quality: Optional[int] = None,
        status: Optional[SignStatus] = None,
        source: Optional[str] = None,
        verified_only: bool = False,
    ) -> list[Sign]:
        """Search signs with multiple filters.

        Args:
            query: Text to search in gloss and english translations
            category: Filter by category
            min_quality: Minimum quality score (1-5)
            status: Filter by status
            source: Filter by source (e.g., "recorded", "wlasl")
            verified_only: Only return verified signs

        Returns:
            List of matching Sign objects
        """
        # Start with all signs or filtered by status
        if verified_only:
            signs = self.store.list_verified()
        elif status:
            signs = self.store.list_signs(status)
        else:
            signs = self.store.list_signs()

        results = []
        for sign in signs:
            if not self._matches_filters(sign, query, category, min_quality, source):
                continue
            results.append(sign)

        return results

    def _matches_filters(
        self,
        sign: Sign,
        query: Optional[str],
        category: Optional[str],
        min_quality: Optional[int],
        source: Optional[str],
    ) -> bool:
        """Check if a sign matches all provided filters."""
        # Query filter (matches gloss or english)
        if query:
            query_lower = query.lower()
            if not self._matches_query(sign, query_lower):
                return False

        # Category filter
        if category and sign.category.lower() != category.lower():
            return False

        # Quality filter
        if min_quality is not None:
            if sign.quality_score is None or sign.quality_score < min_quality:
                return False

        # Source filter
        if source and sign.source.lower() != source.lower():
            return False

        return True

    def _matches_query(self, sign: Sign, query: str) -> bool:
        """Check if sign matches text query."""
        # Check gloss
        if query in sign.gloss.lower():
            return True

        # Check english translations
        for eng in sign.english:
            if query in eng.lower():
                return True

        return False

    def find_by_english(self, word: str) -> list[Sign]:
        """Find signs by English word.

        Args:
            word: English word to search for

        Returns:
            Signs that have this English translation
        """
        word_lower = word.lower()
        results = []

        for sign in self.store.list_verified():
            for eng in sign.english:
                if eng.lower() == word_lower:
                    results.append(sign)
                    break

        return results

    def find_by_category(self, category: str) -> list[Sign]:
        """Find all signs in a category.

        Args:
            category: Category name

        Returns:
            Signs in this category
        """
        return self.search(category=category, verified_only=True)

    def get_categories(self) -> list[str]:
        """Get all unique categories in verified signs."""
        categories = set()
        for sign in self.store.list_verified():
            if sign.category:
                categories.add(sign.category)
        return sorted(categories)

    def get_statistics(self) -> dict:
        """Get database statistics.

        Returns:
            Dict with counts and breakdowns
        """
        verified = self.store.list_verified()
        pending = self.store.list_pending()

        # Quality distribution
        quality_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for sign in verified:
            if sign.quality_score:
                quality_dist[sign.quality_score] += 1

        # Category breakdown
        category_counts = {}
        for sign in verified:
            cat = sign.category or "uncategorized"
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Source breakdown
        source_counts = {}
        for sign in verified:
            source_counts[sign.source] = source_counts.get(sign.source, 0) + 1

        return {
            "total_verified": len(verified),
            "total_pending": len(pending),
            "quality_distribution": quality_dist,
            "categories": category_counts,
            "sources": source_counts,
        }
