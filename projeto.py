import streamlit as st
import google.generativeai as genai
import pdfplumber
import io
import json
from fpdf import FPDF  # Para gerar o PDF


# --- Função para Gerar PDF (Modificada) ---
class PDF(FPDF):
    def header(self):
        pass  # Sem cabeçalho padrão

    def footer(self):
        pass  # Sem rodapé padrão

    def chapter_body(self, body_text):
        # Usando uma fonte padrão do PDF como Arial.
        # FPDF2 tentará lidar com a codificação para caracteres comuns do português.
        self.set_font("Arial", "", 11)
        # O texto 'body_text' é uma string Unicode Python.
        # FPDF2 tentará codificá-la para uma codificação compatível com a fonte (ex: latin-1/cp1252).
        self.multi_cell(
            0, 7, body_text
        )  # Ajuste a altura da linha (7) conforme necessário
        self.ln()


def criar_pdf_cv(texto_cv_revisado, nome_arquivo="cv_melhorado.pdf"):
    """Cria um arquivo PDF a partir do texto do currículo revisado usando fontes padrão."""
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_body(texto_cv_revisado)

    try:
        # Ao gerar para uma string (dest='S'), FPDF retorna uma string codificada em latin-1
        # que representa o arquivo PDF. Isso é para a estrutura do PDF.
        pdf_output_bytes = pdf.output(dest="S").encode("latin-1")
        return pdf_output_bytes, None
    except Exception as e:
        # Embora menos provável com caracteres comuns, um erro ainda pode ocorrer.
        return None, f"Erro ao gerar o PDF: {e}"


# --- Funções Auxiliares (Mesmas da versão anterior) ---


def extrair_texto_pdf(arquivo_pdf_bytes):
    """Extrai texto de um arquivo PDF fornecido como bytes."""
    texto = ""
    try:
        with pdfplumber.open(io.BytesIO(arquivo_pdf_bytes)) as pdf:
            for pagina in pdf.pages:
                texto_pagina = pagina.extract_text()
                if texto_pagina:
                    texto += texto_pagina + "\n"
        if not texto:
            return (
                None,
                "Não foi possível extrair texto do PDF. O PDF pode estar vazio, ser baseado em imagem ou corrompido.",
            )
        return texto, None
    except Exception as e:
        return None, f"Erro ao processar o PDF: {e}"


def analisar_e_revisar_cv_com_gemini(texto_cv, api_key):
    """
    Analisa o texto do CV, fornece sugestões e uma versão revisada
    usando a API Gemini, esperando uma resposta JSON.
    """
    try:
        genai.configure(api_key=api_key)

        generation_config = {
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
        ]

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        prompt_completo = f"""
        Você é um especialista em recrutamento e otimização de currículos (CV) com vasta experiência no mercado de trabalho brasileiro.
        Sua tarefa é analisar o texto do currículo fornecido, oferecer sugestões detalhadas e, MAIS IMPORTANTE, fornecer uma versão completamente revisada e aprimorada do currículo.

        Analise o seguinte currículo:
        ---
        {texto_cv}
        ---

        Por favor, retorne sua resposta ESTRITAMENTE no seguinte formato JSON:
        {{
          "sugestoes_detalhadas": "Aqui você deve fornecer uma análise crítica e sugestões de melhoria para o currículo original. \nExemplos:\n- Erros de português (com correção).\n- Sugestões de palavras-chave.\n- Melhorias na clareza e impacto das descrições (com exemplos).\n- Comentários sobre estrutura e formato (baseado no texto).",
          "curriculo_revisado_texto_completo": "Aqui você deve colocar o TEXTO COMPLETO do currículo, já revisado e com todas as melhorias aplicadas. \nEste texto deve estar pronto para ser copiado e colado em um documento. \nMantenha uma formatação limpa, usando quebras de linha para parágrafos e seções. \nExemplo de estrutura textual:\n\n[Seu Nome Completo]\n[Seu Email] | [Seu Telefone] | [Seu LinkedIn (opcional)]\n\nRESUMO PROFISSIONAL\n[Texto do resumo aqui]...\n\nEXPERIÊNCIA PROFISSIONAL\n[Cargo mais recente]\n[Nome da Empresa] | [Período]\n- [Responsabilidade/Conquista 1]\n- [Responsabilidade/Conquista 2]\n\n[Cargo anterior]..."
        }}

        Certifique-se de que o JSON seja válido. O 'curriculo_revisado_texto_completo' é crucial.
        """
        response = model.generate_content(prompt_completo)

        try:
            cleaned_response_text = response.text.strip()
            if cleaned_response_text.startswith("```json"):
                cleaned_response_text = cleaned_response_text[7:]
            if cleaned_response_text.endswith("```"):
                cleaned_response_text = cleaned_response_text[:-3]

            parsed_json = json.loads(cleaned_response_text)
            return parsed_json, None
        except json.JSONDecodeError as json_e:
            st.error(f"Erro ao decodificar JSON da Gemini: {json_e}")
            st.text("Resposta recebida da Gemini (que causou o erro de JSON):")
            st.text(response.text)
            return (
                None,
                f"A API Gemini não retornou um JSON válido. Detalhes: {response.text}",
            )
        except Exception as e_resp:
            st.error(f"Erro ao processar resposta da Gemini: {e_resp}")
            st.text_area(
                "Resposta bruta da Gemini:",
                response.text if hasattr(response, "text") else str(response),
                height=150,
            )
            return None, f"Erro inesperado ao processar a resposta da API: {e_resp}"

    except Exception as e:
        return (
            None,
            f"Erro ao contatar a API do Gemini: {e}. Verifique sua chave de API e conexão.",
        )


# --- Interface do Streamlit (Modificada para remover aviso da fonte) ---

st.set_page_config(page_title="AutoCV Pro 🚀📄", layout="wide")

st.title("🚀 AutoCV Pro: Analisador e Gerador de Currículos com IA")
st.markdown(
    "Faça o upload do seu currículo em PDF, receba sugestões e gere uma versão melhorada em PDF!"
)

# Entrada da API Key
st.sidebar.header("🔑 Configuração")
api_key_gemini = st.sidebar.text_input(
    "Sua Chave da API Gemini:",
    type="password",
    help="Insira sua chave da API do Google Gemini AI Studio.",
)

# Seção de Upload
st.sidebar.markdown("---")
st.sidebar.header("📄 Faça o Upload do seu CV")
arquivo_pdf = st.sidebar.file_uploader(
    "Selecione o arquivo PDF do seu currículo:", type="pdf"
)

# Botão de Análise
analisar_botao = st.sidebar.button(
    "Analisar e Sugerir Melhorias", disabled=(not api_key_gemini or not arquivo_pdf)
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Nota sobre PDF: A geração do PDF utiliza fontes padrão. A maioria dos caracteres do português é suportada."
)  # Aviso modificado
st.sidebar.markdown("Desenvolvido para a disciplina de Inteligência Artificial.")

# Área Principal para Resultados
col1, col2 = st.columns(2)

with col1:
    st.header("🔍 Análise e Sugestões da IA")
    if "sugestoes_ia" not in st.session_state:
        st.session_state.sugestoes_ia = ""
    if "cv_revisado_texto" not in st.session_state:
        st.session_state.cv_revisado_texto = ""
    if "nome_arquivo_original" not in st.session_state:
        st.session_state.nome_arquivo_original = "cv_original"

    if analisar_botao:
        if not api_key_gemini:
            st.error("Por favor, insira sua chave da API Gemini na barra lateral.")
        elif not arquivo_pdf:
            st.error("Por favor, faça o upload de um arquivo PDF.")
        else:
            st.session_state.nome_arquivo_original = arquivo_pdf.name.replace(
                ".pdf", ""
            )
            with st.spinner("Extraindo texto do PDF... 📄"):
                bytes_pdf = arquivo_pdf.read()
                texto_cv_original, erro_pdf = extrair_texto_pdf(bytes_pdf)

            if erro_pdf:
                st.error(f"Falha na leitura do PDF: {erro_pdf}")
                st.session_state.sugestoes_ia = ""
                st.session_state.cv_revisado_texto = ""
            elif not texto_cv_original or texto_cv_original.strip() == "":
                st.error(
                    "O PDF parece estar vazio ou não contém texto extraível. Tente outro arquivo."
                )
                st.session_state.sugestoes_ia = ""
                st.session_state.cv_revisado_texto = ""
            else:
                with st.spinner(
                    "IA trabalhando... Analisando e revisando seu currículo... 🧠 (Isso pode levar alguns instantes)"
                ):
                    resposta_gemini, erro_gemini = analisar_e_revisar_cv_com_gemini(
                        texto_cv_original, api_key_gemini
                    )

                if erro_gemini:
                    st.error(f"Falha na análise com Gemini: {erro_gemini}")
                    st.session_state.sugestoes_ia = ""
                    st.session_state.cv_revisado_texto = ""
                elif resposta_gemini and isinstance(resposta_gemini, dict):
                    st.session_state.sugestoes_ia = resposta_gemini.get(
                        "sugestoes_detalhadas", "Nenhuma sugestão detalhada fornecida."
                    )
                    st.session_state.cv_revisado_texto = resposta_gemini.get(
                        "curriculo_revisado_texto_completo", ""
                    )
                    st.success("Análise concluída!")
                else:
                    st.warning(
                        "Não foram retornadas sugestões ou o formato da resposta da IA é inesperado."
                    )
                    st.session_state.sugestoes_ia = ""
                    st.session_state.cv_revisado_texto = ""

    if st.session_state.sugestoes_ia:
        st.markdown(st.session_state.sugestoes_ia)
    elif not analisar_botao:
        st.info(
            "⬅️ Configure sua chave da API, faça o upload do seu CV e clique em 'Analisar' para ver as sugestões aqui."
        )


with col2:
    st.header("📄 Currículo Revisado pela IA")
    if st.session_state.cv_revisado_texto:
        st.text_area(
            "Texto do Currículo Revisado:",
            st.session_state.cv_revisado_texto,
            height=400,
        )

        pdf_bytes, erro_pdf_gen = criar_pdf_cv(st.session_state.cv_revisado_texto)
        if erro_pdf_gen:
            st.error(f"Erro ao preparar PDF para download: {erro_pdf_gen}")
        elif pdf_bytes:
            nome_pdf_download = (
                f"{st.session_state.nome_arquivo_original}_AutoCV_melhorado.pdf"
            )
            st.download_button(
                label="📥 Baixar Currículo Melhorado em PDF",
                data=pdf_bytes,
                file_name=nome_pdf_download,
                mime="application/pdf",
            )
    else:
        st.info(
            "O texto do currículo revisado pela IA aparecerá aqui após a análise. Você poderá então baixá-lo em PDF."
        )
