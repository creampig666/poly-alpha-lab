import pytest

from poly_alpha_lab.models import Market, MarketStructureError, OrderBook, OrderLevel


def test_market_parses_gamma_json_string_fields_and_maps_yes_no() -> None:
    market = Market.model_validate(
        {
            "id": 123,
            "question": "Will BTC hit 100k?",
            "active": "true",
            "closed": "false",
            "enableOrderBook": "true",
            "liquidity": "1234.5",
            "outcomes": '["Yes","No"]',
            "outcomePrices": '["0.25","0.75"]',
            "clobTokenIds": '["yes-token","no-token"]',
        }
    )

    assert market.id == "123"
    assert market.active is True
    assert market.closed is False
    assert market.enable_order_book is True
    assert market.liquidity == 1234.5
    assert market.outcomes == ["Yes", "No"]
    assert market.outcome_prices == [0.25, 0.75]
    assert market.clob_token_ids == ["yes-token", "no-token"]
    assert market.yes_outcome_index == 0
    assert market.no_outcome_index == 1
    assert market.yes_price == 0.25
    assert market.no_price == 0.75
    assert market.yes_token_id == "yes-token"
    assert market.no_token_id == "no-token"


def test_market_parses_gamma_list_fields_and_maps_reversed_yes_no() -> None:
    market = Market.model_validate(
        {
            "id": "m1",
            "outcomes": ["No", "Yes"],
            "outcomePrices": [0.62, 0.38],
            "clobTokenIds": ["no-token", "yes-token"],
        }
    )

    assert market.yes_outcome_index == 1
    assert market.no_outcome_index == 0
    assert market.yes_price == 0.38
    assert market.no_price == 0.62
    assert market.yes_token_id == "yes-token"
    assert market.no_token_id == "no-token"


def test_market_raises_clear_error_when_yes_no_or_token_missing() -> None:
    market = Market.model_validate(
        {
            "id": "bad",
            "outcomes": ["Up", "Down"],
            "outcomePrices": [0.5, 0.5],
            "clobTokenIds": ["a", "b"],
        }
    )

    with pytest.raises(MarketStructureError, match="missing 'Yes'"):
        _ = market.yes_outcome_index

    empty_token = Market.model_validate(
        {
            "id": "empty-token",
            "outcomes": ["Yes", "No"],
            "outcomePrices": [0.5, 0.5],
            "clobTokenIds": ["yes-token", ""],
        }
    )
    with pytest.raises(MarketStructureError, match="empty NO CLOB token id"):
        _ = empty_token.no_token_id


def test_outcome_prices_length_mismatch_raises_market_structure_error() -> None:
    with pytest.raises(MarketStructureError, match="outcomePrices length 1"):
        Market.model_validate(
            {
                "id": "bad-prices",
                "outcomes": ["Yes", "No"],
                "outcomePrices": [0.5],
                "clobTokenIds": ["yes-token", "no-token"],
            }
        )


def test_clob_token_ids_length_mismatch_raises_market_structure_error() -> None:
    with pytest.raises(MarketStructureError, match="clobTokenIds length 1"):
        Market.model_validate(
            {
                "id": "bad-tokens",
                "outcomes": ["Yes", "No"],
                "outcomePrices": [0.5, 0.5],
                "clobTokenIds": ["yes-token"],
            }
        )


def test_orderbook_parses_clob_response_and_sorts_levels() -> None:
    book = OrderBook.model_validate(
        {
            "asset_id": "token-1",
            "market": "condition-1",
            "bids": [{"price": "0.4", "size": "1"}, {"price": "0.5", "size": "1"}],
            "asks": [{"price": "0.6", "size": "1"}, {"price": "0.55", "size": "1"}],
        }
    )

    assert book.token_id == "token-1"
    assert [level.price for level in book.bids] == [0.5, 0.4]
    assert [level.price for level in book.asks] == [0.55, 0.6]
