# app/tools/card.py
from semantic_kernel.functions import kernel_function
from app.knowledge_base import get_card_recommendation, search_card_benefits
import logging

logger = logging.getLogger(__name__)

class CardTools:
    @kernel_function(name="recommend_card", description="Recommend the best credit card for a spending category and country. Categories: dining, travel, hotels, shopping, general")
    def recommend_card(self, category: str, country: str, amount: float = 100.0):
        """
        Recommend the best credit card for a spending category and country.

        Args:
            category: Spending category (dining, travel, hotels, shopping, general)
            country: Country where transaction will occur
            amount: Transaction amount (default 100.0)

        Returns:
            Card recommendation with benefits and explanation
        """
        try:
            logger.info(f"⚡️ Card tool called: category={category}, country={country}, amount=${amount}")

            # Normalize category input
            category_map = {
                "restaurant": "dining",
                "restaurants": "dining",
                "food": "dining",
                "hotel": "travel",
                "hotels": "travel",
                "flight": "travel",
                "flights": "travel",
                "airline": "travel",
                "general": "travel",  # Default to travel for this travel concierge app
            }

            category = category_map.get(category.lower(), category.lower())

            # Get recommendations for different card types
            cards_to_check = ["BankGold", "BankPlatinum", "BankRewards"]
            best_card = None
            best_score = 0

            for card in cards_to_check:
                recommendation = get_card_recommendation(
                    card, category, amount, country
                )

                points = recommendation.get("points_earned", 0)
                fx_fee = recommendation.get("fx_fee", "Unknown")

                score = points
                if "None" in fx_fee or "0%" in fx_fee:
                    score += 10
                elif fx_fee != "Unknown":
                    score -= 5

                if score > best_score:
                    best_score = score
                    best_card = recommendation

            if best_card is None:
                raise LookupError(f"No card recommendation found for {category}")

            return {
                "best": best_card,
                "explanation": (
                    f"Recommended {best_card['card']} for {category} spending. "
                    f"{best_card['recommendation']}"
                ),
            }

        except Exception as e:
            logger.error(f"❌ Failed to get card recommendation: {e}")
            raise
