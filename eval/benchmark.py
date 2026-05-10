from __future__ import annotations


class BenchmarkHarness:
    def on_prefill_start(self) -> None:
        return None

    def on_prefill_end(self) -> None:
        return None

    def on_decode_start(self) -> None:
        return None

    def on_decode_token_end(self, token_index: int) -> None:
        return None

    def on_layer_copy_start(self, layer_id: int) -> None:
        return None

    def on_layer_copy_end(self, layer_id: int) -> None:
        return None

    def on_layer_compute_start(self, layer_id: int) -> None:
        return None

    def on_layer_compute_end(self, layer_id: int) -> None:
        return None

    def get_summary(self) -> dict[str, float]:
        raise NotImplementedError

    def export_csv(self, output_path: str) -> None:
        raise NotImplementedError
