"""
nanobanana MCPサーバー

GoogleのGemini 2.5 Flash Image APIをClaude Codeで使用できるModel Context Protocol (MCP) サーバーです。
"""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Sequence, Union

from fastmcp import FastMCP
from pydantic import BaseModel

from .config import get_settings, setup_logging
from .constants import PROJECT_NAME, PROJECT_VERSION, MCP_VERSION
from .gemini_client import create_gemini_client, get_gemini_client
from .tools import generate, edit, blend, status
from .models.schemas import create_error_response

# 設定とロギング初期化
settings = get_settings()
setup_logging(settings)
logger = logging.getLogger(__name__)


# ================================
# Lifespanコンテキストマネージャー
# ================================

@asynccontextmanager
async def lifespan(mcp):
    """FastMCPライフサイクル管理 - startup/shutdownをFastMCPイベントループ内で実行"""
    # Startup
    try:
        logger.info("Starting nanobanana-mcp MCP Server...")

        # Geminiクライアント初期化
        gemini_client = await create_gemini_client()
        logger.info("Gemini client initialized successfully")

        # 出力ディレクトリの確認と作成
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ready: {settings.output_dir}")

        logger.info("Server startup completed successfully")

    except Exception as e:
        logger.error(f"Server startup error: {e}")
        raise

    yield  # サーバー実行

    # Shutdown
    try:
        logger.info("Shutting down Nanobanana MCP Server...")

        # 統計情報をログ
        try:
            gemini_client = get_gemini_client()
            stats = gemini_client.get_statistics()
            logger.info(f"Session statistics: {stats}")
        except Exception as e:
            logger.warning(f"Could not retrieve session statistics: {e}")

        # 最終クリーンアップ
        if settings.dev_mode:
            try:
                from .utils.file_manager import get_file_manager
                file_manager = get_file_manager()
                cache_result = file_manager.manage_cache()
                logger.info(f"Cache management: {cache_result}")
            except Exception as e:
                logger.warning(f"Cache management failed: {e}")

        logger.info("Nanobanana MCP Server shut down gracefully")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# FastMCPサーバーインスタンス生成 (lifespan含む)
mcp_server = FastMCP(
    name=settings.server_name,
    version=settings.server_version,
    lifespan=lifespan
)


# ================================
# MCPツール登録
# ================================

@mcp_server.tool()
async def nanobanana_generate(
    prompt: str,
    aspect_ratio: Optional[str] = None,
    style: Optional[str] = None,
    quality: Optional[str] = "high",
    output_format: Optional[str] = "png",
    candidate_count: Optional[Union[int, str]] = 1,
    additional_keywords: Optional[List[str]] = None,
    optimize_prompt: Optional[Union[bool, str]] = True
) -> Dict[str, Any]:
    """Generate images from text prompts using Gemini 2.5 Flash Image"""
    try:
        return await generate.nanobanana_generate(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            style=style,
            quality=quality,
            output_format=output_format,
            candidate_count=candidate_count,
            additional_keywords=additional_keywords,
            optimize_prompt=optimize_prompt
        )
    except Exception as e:
        logger.error(f"Error in nanobanana_generate: {e}")
        return create_error_response(
            f"Generation failed: {str(e)}",
            "GENERATION_ERROR"
        ).dict()


@mcp_server.tool()
async def nanobanana_edit(
    image_path: str,
    edit_prompt: str,
    mask_path: Optional[str] = None,
    output_format: Optional[str] = "png",
    quality: Optional[str] = "high",
    optimize_prompt: Optional[Union[bool, str]] = True
) -> Dict[str, Any]:
    """Edit existing images with natural language instructions"""
    try:
        return await edit.nanobanana_edit(
            image_path=image_path,
            edit_prompt=edit_prompt,
            mask_path=mask_path,
            output_format=output_format,
            quality=quality,
            optimize_prompt=optimize_prompt
        )
    except Exception as e:
        logger.error(f"Error in nanobanana_edit: {e}")
        return create_error_response(
            f"Edit failed: {str(e)}",
            "EDIT_ERROR"
        ).dict()


@mcp_server.tool()
async def nanobanana_blend(
    image_paths: List[str],
    blend_prompt: str,
    maintain_consistency: Optional[Union[bool, str]] = True,
    output_format: Optional[str] = "png",
    quality: Optional[str] = "high",
    optimize_prompt: Optional[Union[bool, str]] = True
) -> Dict[str, Any]:
    """Blend multiple images into a new composition"""
    try:
        return await blend.nanobanana_blend(
            image_paths=image_paths,
            blend_prompt=blend_prompt,
            maintain_consistency=maintain_consistency,
            output_format=output_format,
            quality=quality,
            optimize_prompt=optimize_prompt
        )
    except Exception as e:
        logger.error(f"Error in nanobanana_blend: {e}")
        return create_error_response(
            f"Blend failed: {str(e)}",
            "BLEND_ERROR"
        ).dict()


@mcp_server.tool()
async def nanobanana_status(
    detailed: Optional[Union[bool, str]] = True,
    include_history: Optional[Union[bool, str]] = False,
    reset_stats: Optional[Union[bool, str]] = False
) -> Dict[str, Any]:
    """Check server status and API connectivity"""
    try:
        return await status.nanobanana_status(
            detailed=detailed,
            include_history=include_history,
            reset_stats=reset_stats
        )
    except Exception as e:
        logger.error(f"Error in nanobanana_status: {e}")
        return create_error_response(
            f"Status check failed: {str(e)}",
            "STATUS_ERROR"
        ).dict()


# ================================
# MCPリソース (オプション)
# ================================

class ServerInfoResource(BaseModel):
    """サーバー情報リソース"""
    name: str
    version: str
    mcp_version: str
    description: str
    tools: List[str]


@mcp_server.resource("server://info")
async def get_server_info() -> ServerInfoResource:
    """サーバー情報リソースを提供"""
    return ServerInfoResource(
        name=PROJECT_NAME,
        version=PROJECT_VERSION,
        mcp_version=MCP_VERSION,
        description="Gemini 2.5 Flash Image MCP Server for Claude Code",
        tools=[
            "nanobanana_generate",
            "nanobanana_edit", 
            "nanobanana_blend",
            "nanobanana_status"
        ]
    )


# ================================
# シグナルハンドリング (グレースフルシャットダウン)
# ================================

def signal_handler(signum: int, frame) -> None:
    """シグナルハンドラー (Ctrl+C等)

    Note: 実際のcleanupはlifespanコンテキストマネージャーで処理
    """
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    sys.exit(0)


# ================================
# サーバー実行関数
# ================================

async def run_server_async():
    """非同期サーバー実行

    Note: startup/shutdownはlifespanコンテキストマネージャーで処理
    """
    try:
        # シグナルハンドラー登録 (Unix系のみ)
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except AttributeError:
            # Windowsでは一部シグナルがサポートされない場合あり
            logger.warning("Some signals not supported on this platform")

        # サーバー実行 (lifespanがstartup/shutdownを処理)
        if settings.dev_mode:
            logger.info("Server running in stdio mode for MCP")
            await mcp_server.run(transport="stdio")
        else:
            logger.info(f"Server listening on {settings.host}:{settings.port}")
            await mcp_server.run(
                host=settings.host,
                port=settings.port,
                transport="websocket"
            )

    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


def run_server():
    """同期サーバー実行 (メインエントリーポイント)"""
    try:
        logger.info(f"Starting {PROJECT_NAME} MCP Server...")

        # MCP stdioモードではFastMCPがイベントループを直接管理
        if settings.dev_mode:
            logger.info("Starting MCP server in stdio mode...")
            # 同期的開始 - FastMCPが内部でイベントループを生成
            setup_and_run_mcp_sync()
        else:
            # WebSocketモードでは既存方式を維持
            logger.info("Starting WebSocket mode...")
            import asyncio
            asyncio.run(run_server_async())
            
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal server error: {e}")
        sys.exit(1)


def setup_and_run_mcp_sync():
    """MCP stdioモード用の同期的設定と実行

    Note: startup/shutdownはlifespanコンテキストマネージャーで処理
    """
    try:
        logger.info("Initializing MCP server synchronously...")
        logger.info("Server running in stdio mode for MCP")

        # FastMCPが独自イベントループを生成しlifespanコンテキストを管理
        mcp_server.run(transport="stdio")

    except Exception as e:
        logger.error(f"MCP setup error: {e}")
        raise


# ================================
# CLIインターフェース
# ================================

def main():
    """CLIメイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_NAME} - Gemini 2.5 Flash Image MCP Server"
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"{PROJECT_NAME} {PROJECT_VERSION}"
    )
    
    parser.add_argument(
        "--host",
        default=settings.host,
        help=f"Host to bind to (default: {settings.host})"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help=f"Port to bind to (default: {settings.port})"
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        default=settings.dev_mode,
        help="Run in development mode"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        default=settings.debug,
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--check-health",
        action="store_true",
        help="Check API health and exit"
    )
    
    parser.add_argument(
        "--reset-stats",
        action="store_true", 
        help="Reset server statistics"
    )
    
    args = parser.parse_args()

    # 設定オーバーライド
    if args.host != settings.host:
        settings.host = args.host
    if args.port != settings.port:
        settings.port = args.port
    if args.dev:
        settings.dev_mode = True
    if args.debug:
        settings.debug = True
        # ロギングレベル更新
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # 特殊コマンド処理
    if args.check_health:
        asyncio.run(check_health_and_exit())
        return

    if args.reset_stats:
        asyncio.run(reset_stats_and_exit())
        return

    # サーバー起動
    logger.info(f"Configuration: host={settings.host}, port={settings.port}, dev={settings.dev_mode}")

    # MCPモード検知 (Claude Codeで -m src.server として実行時)
    if len(sys.argv) == 1 and not sys.stdin.isatty():
        # stdinがターミナルでなければMCPモードとみなす
        run_mcp_server()
    else:
        run_server()


async def check_health_and_exit():
    """APIステータス確認後終了"""
    try:
        print(f"Checking {PROJECT_NAME} health...")

        # Geminiクライアント生成とテスト
        gemini_client = await create_gemini_client(settings)
        health = await gemini_client.health_check()
        
        print(f"API Status: {health['status']}")
        print(f"Model: {health.get('model', 'unknown')}")
        print(f"Accessible: {health.get('api_accessible', False)}")
        
        if health.get('error'):
            print(f"Error: {health['error']}")
            sys.exit(1)
        else:
            print("✅ Health check passed")
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        sys.exit(1)


async def reset_stats_and_exit():
    """統計リセット後終了"""
    try:
        print("Resetting server statistics...")
        
        gemini_client = await create_gemini_client(settings)
        gemini_client.reset_statistics()
        
        print("✅ Statistics reset successfully")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Statistics reset failed: {e}")
        sys.exit(1)


# ================================
# 開発用ヘルパー関数
# ================================

def get_server_info() -> Dict[str, Any]:
    """サーバー情報を返す (同期関数)"""
    return {
        "name": PROJECT_NAME,
        "version": PROJECT_VERSION,
        "mcp_version": MCP_VERSION,
        "settings": {
            "host": settings.host,
            "port": settings.port,
            "dev_mode": settings.dev_mode,
            "debug": settings.debug
        },
        "tools": [
            "nanobanana_generate",
            "nanobanana_edit",
            "nanobanana_blend", 
            "nanobanana_status"
        ]
    }


def list_available_tools() -> List[Dict[str, Any]]:
    """利用可能なツールリストを返す"""
    return [
        generate.TOOL_METADATA,
        edit.TOOL_METADATA,
        blend.TOOL_METADATA,
        status.TOOL_METADATA
    ]


# ================================
# エントリーポイント
# ================================

# FastMCPサーバー実行用のシンプルな関数
def run_mcp_server():
    """MCPサーバー実行 (Claude Codeから呼び出し)"""
    logger.info("Starting nanobanana-mcp in MCP mode...")

    # デバッグ: 環境変数を確認
    import os
    api_keys = ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_API_KEY"]
    for key in api_keys:
        val = os.environ.get(key, "")
        if val:
            logger.info(f"🔍 DEBUG: Found {key} in os.environ: {val[:10]}...")
        else:
            logger.info(f"🔍 DEBUG: {key} not found in os.environ")

    # APIキー検証のための簡単な初期確認
    try:
        from .config_keyloader import SecureKeyLoader

        # キーローダーでAPIキー確認
        key_loader = SecureKeyLoader(mcp_server_name="nanobanana")
        
        if not key_loader.has_key():
            logger.error(
                "Gemini API key not found. Please set it in:\n"
                "1. MCP server configuration (recommended): mcpServers.nanobanana.env.GEMINI_API_KEY\n"
                "2. .env file: GEMINI_API_KEY, GOOGLE_API_KEY, or GOOGLE_AI_API_KEY"
            )
            return
        else:
            debug_info = key_loader.get_debug_info()
            logger.info(f"🔐 API key loaded from: {debug_info['key_info']['source_name']}")
            logger.info(f"🔐 Key name: {debug_info['key_info']['key_name']}")
            
            # 環境変数汚染検証
            pollution_check = key_loader.verify_no_os_env_pollution()
            logger.info(f"🔐 {pollution_check['message']}")

    except Exception as e:
        logger.warning(f"Key validation warning: {e}")
        logger.info("Proceeding with server startup (key will be validated during first use)")

    # stdioモードでサーバー実行
    mcp_server.run(transport="stdio")

if __name__ == "__main__":
    # MCPモード検知: stdinがTTYでなければMCP stdioモード
    import sys

    if not sys.stdin.isatty():
        # MCPモード: stdinがパイプされている (Claude Codeから呼び出し)
        logger.info("Detected MCP mode (stdio transport)")
        run_mcp_server()
    else:
        # CLIモード: ターミナルから直接実行
        logger.info("Detected CLI mode")
        main()
else:
    # モジュールとしてimport時は自動実行しない
    pass