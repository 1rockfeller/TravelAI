# app/tools/fx.py
from semantic_kernel.functions import kernel_function
import requests

class FxTools:
    @kernel_function(name="convert_fx", description="Convert currency via Frankfurter")
    def convert_fx(self, amount: float, base: str, target: str):
        """
        Convert currency using the Frankfurter API.

        Args:
            amount: Amount to convert
            base: Source currency code (e.g., USD)
            target: Target currency code (e.g., EUR)

        Returns:
            Frankfurter API response with the converted amount and rate.
        """
        if amount < 0:
            raise ValueError("Amount must be non-negative")
        base = base.strip().upper()
        target = target.strip().upper()
        if len(base) != 3 or len(target) != 3 or not base.isalpha() or not target.isalpha():
            raise ValueError("Currency codes must be three-letter ISO codes")

        response = requests.get(
            "https://api.frankfurter.app/latest",
            params={"amount": amount, "from": base, "to": target},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
