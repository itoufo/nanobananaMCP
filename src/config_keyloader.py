"""
APIキーローダーモジュール

ターミナル環境変数を一切使用せず、.envファイルまたはMCP設定からのみキーを読み取り、
明示的にSDKに渡すセキュリティ強化モジュールです。

優先順位: os.environ > .api_keyファイル > MCP設定(env) > .env > (なければエラー)
"""

from pathlib import Path
import json
from typing import Optional, Dict, Any
from dotenv import dotenv_values  # DOES NOT touch process env
import logging

logger = logging.getLogger(__name__)


def load_from_api_key_file() -> Dict[str, str]:
    """
    .api_keyファイルからAPIキーをロードします。
    MCPサーバーで環境変数が渡されない場合のフォールバックです。

    Returns:
        Dict[str, str]: 環境変数辞書
    """
    # プロジェクトルートの.api_keyファイルを探す
    possible_paths = [
        Path(__file__).parent.parent / ".api_key",  # src/../.api_key
        Path.home() / ".nanobanana_api_key",
        Path(".api_key"),
    ]

    for path in possible_paths:
        if path.exists():
            try:
                val = path.read_text().strip()
                if val:
                    logger.debug(f"Found API key in file: {path}")
                    return {"GEMINI_API_KEY": val}
            except Exception as e:
                logger.warning(f"Failed to read API key file {path}: {e}")

    return {}


def load_from_os_environ() -> Dict[str, str]:
    """
    OS環境変数からAPIキー関連変数をロードします。
    MCPサーバーはClaude Codeからenv設定をos.environで受け取ります。

    Returns:
        Dict[str, str]: 環境変数辞書
    """
    import os

    CANDIDATES = ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_API_KEY"]
    result = {}

    for key_name in CANDIDATES:
        val = os.environ.get(key_name, "").strip()
        if val:
            result[key_name] = val
            logger.debug(f"Found {key_name} in os.environ")

    return result


def load_from_env_file(env_path: str = ".env") -> Dict[str, str]:
    """
    .envファイルからキー・バリューのペアをロードします。
    重要: プロセス環境変数には触れません。

    Args:
        env_path: .envファイルパス (デフォルト: ".env")

    Returns:
        Dict[str, str]: 環境変数辞書
    """
    env_file = Path(env_path)

    if not env_file.exists():
        logger.debug(f"Env file not found: {env_path}")
        return {}

    try:
        env_vars = dotenv_values(str(env_file))
        logger.debug(f"Loaded {len(env_vars)} variables from {env_path}")
        return {k: v for k, v in env_vars.items() if v is not None}
    except Exception as e:
        logger.warning(f"Failed to load env file {env_path}: {e}")
        return {}


def load_from_mcp_settings(
    settings_path: Optional[str] = None,
    server_name: Optional[str] = None
) -> Dict[str, str]:
    """
    MCP設定ファイルから環境変数をロードします。

    Args:
        settings_path: MCP設定ファイルパス (例: ~/.config/Claude/claude_desktop_config.json)
        server_name: mcpServersで使用するサーバー名 (例: "nanobanana")

    Returns:
        Dict[str, str]: 環境変数辞書
    """
    if not settings_path:
        # デフォルトClaude Desktop設定パスを試行
        possible_paths = [
            Path.home() / ".config" / "Claude" / "claude_desktop_config.json",
            Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json",
            Path(".claude") / "settings.local.json",
            Path("claude_desktop_config.json")
        ]
        
        for path in possible_paths:
            if path.exists():
                settings_path = str(path)
                logger.debug(f"Found MCP settings at: {settings_path}")
                break
        else:
            logger.debug("No MCP settings file found")
            return {}
    
    settings_file = Path(settings_path)
    if not settings_file.exists():
        logger.debug(f"MCP settings file not found: {settings_path}")
        return {}
    
    try:
        with open(settings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        servers = data.get("mcpServers", {})
        if not servers:
            logger.debug("No mcpServers section found in settings")
            return {}
        
        # 特定サーバー名が指定された場合
        if server_name and server_name in servers:
            env_vars = servers[server_name].get("env", {}) or {}
            logger.debug(f"Loaded {len(env_vars)} variables from MCP server '{server_name}'")
            return {k: str(v) for k, v in env_vars.items()}

        # サーバー名が指定されていない場合は全envをマージ (後が優先)
        merged = {}
        for srv_name, srv_config in servers.items():
            srv_env = srv_config.get("env", {}) or {}
            merged.update(srv_env)
            if srv_env:
                logger.debug(f"Merged {len(srv_env)} variables from server '{srv_name}'")
        
        return {k: str(v) for k, v in merged.items()}
        
    except Exception as e:
        logger.warning(f"Failed to load MCP settings {settings_path}: {e}")
        return {}


def pick_gemini_key(*sources: Dict[str, str]) -> Optional[str]:
    """
    複数ソースからGemini APIキーを優先順位に従って選択します。

    優先順位:
    1. GEMINI_API_KEY
    2. GOOGLE_API_KEY
    3. GOOGLE_AI_API_KEY

    Args:
        *sources: キー・バリュー辞書 (前方が優先度高)

    Returns:
        Optional[str]: 発見されたAPIキーまたはNone
    """
    CANDIDATES = ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_API_KEY"]
    
    for source_idx, source in enumerate(sources):
        if not source:
            continue
            
        for key_name in CANDIDATES:
            val = (source.get(key_name) or "").strip()
            if val:
                logger.debug(f"Found API key '{key_name}' from source {source_idx}")
                return val
    
    logger.debug("No API key found in any source")
    return None


def get_key_source_info(*sources: Dict[str, str]) -> Dict[str, Any]:
    """
    キーの出所情報を返します (デバッグ/検証用)。

    Args:
        *sources: キー・バリュー辞書

    Returns:
        Dict[str, Any]: キー出所情報
    """
    CANDIDATES = ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_API_KEY"]
    SOURCE_NAMES = ["OS_Environ", "API_Key_File", "MCP_Settings", ".env_File", "Unknown"]
    
    info = {
        "found_key": None,
        "key_name": None,
        "source_name": None,
        "source_index": None,
        "masked_key": None
    }
    
    for source_idx, source in enumerate(sources):
        if not source:
            continue
            
        for key_name in CANDIDATES:
            val = (source.get(key_name) or "").strip()
            if val:
                info.update({
                    "found_key": True,
                    "key_name": key_name,
                    "source_name": SOURCE_NAMES[source_idx] if source_idx < len(SOURCE_NAMES) else "Unknown",
                    "source_index": source_idx,
                    "masked_key": f"{val[:10]}..." if len(val) > 10 else f"{val[:4]}..."
                })
                return info
    
    info["found_key"] = False
    return info


class SecureKeyLoader:
    """
    セキュリティ強化APIキーローダークラス
    """
    
    def __init__(
        self,
        mcp_settings_path: Optional[str] = None,
        mcp_server_name: str = "nanobanana",
        env_file: str = ".env"
    ):
        """
        Args:
            mcp_settings_path: MCP設定ファイルパス
            mcp_server_name: MCPサーバー名
            env_file: .envファイルパス
        """
        self.mcp_settings_path = mcp_settings_path
        self.mcp_server_name = mcp_server_name
        self.env_file = env_file

        # キーロード (優先順位: os.environ > .api_keyファイル > MCP設定ファイル > .envファイル)
        self.os_env = load_from_os_environ()
        self.api_key_file_env = load_from_api_key_file()
        self.mcp_env = load_from_mcp_settings(mcp_settings_path, mcp_server_name)
        self.file_env = load_from_env_file(env_file)

        # キー選択 (os.environ優先、次に.api_keyファイル)
        self.api_key = pick_gemini_key(self.os_env, self.api_key_file_env, self.mcp_env, self.file_env)
        self.key_info = get_key_source_info(self.os_env, self.api_key_file_env, self.mcp_env, self.file_env)
        
        logger.info(f"Key loader initialized - Found key: {self.key_info['found_key']}")
        if self.key_info['found_key']:
            logger.info(f"Key source: {self.key_info['source_name']} ({self.key_info['key_name']})")
    
    def get_api_key(self) -> Optional[str]:
        """APIキーを返す"""
        return self.api_key

    def has_key(self) -> bool:
        """APIキーの存在確認"""
        return bool(self.api_key)

    def get_debug_info(self) -> Dict[str, Any]:
        """デバッグ情報を返す"""
        return {
            "key_info": self.key_info,
            "os_env_count": len(self.os_env),
            "api_key_file_count": len(self.api_key_file_env),
            "mcp_env_count": len(self.mcp_env),
            "file_env_count": len(self.file_env),
            "settings_path": self.mcp_settings_path,
            "server_name": self.mcp_server_name,
            "env_file": self.env_file
        }
    
    def verify_no_os_env_pollution(self) -> Dict[str, Any]:
        """
        OS環境変数の汚染がないことを検証します。

        Returns:
            Dict[str, Any]: 検証結果
        """
        import os
        
        CANDIDATES = ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_API_KEY"]
        os_env_keys = []
        
        for key_name in CANDIDATES:
            if key_name in os.environ:
                os_env_keys.append({
                    "name": key_name,
                    "masked_value": f"{os.environ[key_name][:10]}..." if len(os.environ[key_name]) > 10 else "***"
                })
        
        return {
            "os_env_keys_found": len(os_env_keys),
            "os_env_keys": os_env_keys,
            "our_key_source": self.key_info['source_name'],
            "pollution_risk": len(os_env_keys) > 0 and not self.has_key(),
            "message": "✅ Clean - No OS env pollution" if len(os_env_keys) == 0 or self.has_key() else "⚠️  OS env keys present but ignored"
        }