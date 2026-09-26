"""
Configuração central — cliente OpenRouter e parâmetros da avaliação.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Settings:
    # --- Credenciais e modelos ---
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))

    # Modelo que atua como JUIZ. Vale manter separado do modelo avaliado:
    # usar o mesmo modelo para gerar e julgar introduz autopreferência —
    # G-Eval (Liu et al., 2023) alerta para o viés a favor de textos gerados
    # por LLM, e o risco de auto-reforço quando o juiz vira sinal de recompensa.
    juiz_model: str = field(
        default_factory=lambda: os.getenv("JUIZ_MODEL", "openai/gpt-4o-mini")
    )
    # Modelo avaliado (só usado se você gerar respostas novas por aqui).
    modelo_avaliado: str = field(
        default_factory=lambda: os.getenv("MODELO_AVALIADO", "openai/gpt-4o-mini")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "openai/text-embedding-3-small"
        )
    )

    # --- Geração do juiz ---
    temperature: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.0"))
    )
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "800")))

    # --- G-Eval ---
    # O artigo estima a distribuição dos scores pelas probabilidades dos tokens.
    # Quando a API expõe logprobs, usamos isso direto. Quando não expõe, o
    # próprio artigo amostra n vezes com temperatura 1 (n=20 no paper).
    geval_usar_logprobs: bool = field(
        default_factory=lambda: os.getenv("GEVAL_LOGPROBS", "true").lower() == "true"
    )
    geval_amostras: int = field(
        default_factory=lambda: int(os.getenv("GEVAL_AMOSTRAS", "8"))
    )
    geval_escala_max: int = field(
        default_factory=lambda: int(os.getenv("GEVAL_ESCALA_MAX", "5"))
    )

    # --- RAGAS ---
    # Número de perguntas reversas geradas para a relevância da resposta
    # (n na Eq. 1 do artigo).
    ragas_n_perguntas: int = field(
        default_factory=lambda: int(os.getenv("RAGAS_N_PERGUNTAS", "3"))
    )

    # --- PPI (ARES) ---
    # Nível de confiança dos intervalos. O ARES usa alpha 0.05 (95%).
    ppi_alpha: float = field(
        default_factory=lambda: float(os.getenv("PPI_ALPHA", "0.05"))
    )

    @property
    def tem_chave(self) -> bool:
        return bool(self.api_key)

    @property
    def juiz_e_avaliado_iguais(self) -> bool:
        """Sinaliza risco de autopreferência."""
        return self.juiz_model == self.modelo_avaliado


SETTINGS = Settings()


def get_client(settings: Settings | None = None) -> OpenAI:
    settings = settings or SETTINGS
    if not settings.api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY não encontrada. Copie .env.example para .env e "
            "preencha a chave. As métricas lexicais, a concordância, o PPI e os "
            "testes rodam sem chave."
        )
    return OpenAI(
        api_key=settings.api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/jgamacyber",
            "X-Title": "BLIS-Evals",
        },
    )


def resumo_config(settings: Settings | None = None) -> str:
    s = settings or SETTINGS
    aviso = "  [!] juiz == avaliado" if s.juiz_e_avaliado_iguais else ""
    return (
        f"juiz={s.juiz_model} | avaliado={s.modelo_avaliado} | "
        f"temp={s.temperature} | api_key={'OK' if s.tem_chave else 'AUSENTE'}{aviso}"
    )
