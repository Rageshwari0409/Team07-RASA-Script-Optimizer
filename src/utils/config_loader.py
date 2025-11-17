"""Configuration loader utility."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


class ConfigLoader:
    """Load and manage configuration from YAML files and environment variables."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config: Dict[str, Any] = {}
        self.prompts: Dict[str, Any] = {}
        
        # Load environment variables
        env_file = self.config_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
        
        # Load configuration files
        self._load_config()
        self._load_prompts()
        self._override_with_env()
    
    def _load_config(self):
        """Load main configuration from YAML file."""
        config_file = self.config_dir / "config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    def _load_prompts(self):
        """Load prompts configuration from YAML file."""
        prompts_file = self.config_dir / "prompts.yaml"
        if prompts_file.exists():
            with open(prompts_file, 'r') as f:
                self.prompts = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    def _override_with_env(self):
        """Override configuration with environment variables."""
        # Azure OpenAI
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            self.config['azure_openai']['endpoint'] = os.getenv("AZURE_OPENAI_ENDPOINT")
        if os.getenv("AZURE_OPENAI_API_KEY"):
            self.config['azure_openai']['api_key'] = os.getenv("AZURE_OPENAI_API_KEY")
        if os.getenv("AZURE_OPENAI_API_VERSION"):
            self.config['azure_openai']['api_version'] = os.getenv("AZURE_OPENAI_API_VERSION")
        if os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"):
            self.config['azure_openai']['deployment_name'] = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"):
            self.config['embeddings']['deployment_name'] = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

        # Milvus
        if os.getenv("MILVUS_HOST"):
            self.config['milvus']['host'] = os.getenv("MILVUS_HOST")
        if os.getenv("MILVUS_PORT"):
            self.config['milvus']['port'] = int(os.getenv("MILVUS_PORT"))
        if os.getenv("MILVUS_USER"):
            self.config['milvus']['user'] = os.getenv("MILVUS_USER")
        if os.getenv("MILVUS_PASSWORD"):
            self.config['milvus']['password'] = os.getenv("MILVUS_PASSWORD")
        if os.getenv("MILVUS_SECURE"):
            self.config['milvus']['secure'] = os.getenv("MILVUS_SECURE").lower() == 'true'
        if os.getenv("MILVUS_COLLECTION_NAME"):
            self.config['milvus']['collection_name'] = os.getenv("MILVUS_COLLECTION_NAME")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'azure_openai.endpoint')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_prompt(self, prompt_name: str) -> str:
        """Get prompt template by name.
        
        Args:
            prompt_name: Name of the prompt template
            
        Returns:
            Prompt template string
        """
        return self.prompts.get(prompt_name, "")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration.
        
        Returns:
            Complete configuration dictionary
        """
        return self.config


# Global configuration instance
_config_instance = None


def get_config() -> ConfigLoader:
    """Get global configuration instance.
    
    Returns:
        ConfigLoader instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader()
    return _config_instance

