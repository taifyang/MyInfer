#pragma once

#include <string>
#include <utility>
#include <vector>
#include <memory>

namespace my_infer 
{
	// 词语的类型
	enum class TokenType 
	{
		TokenUnknown = -9,
		TokenInputNumber = -8,
		TokenComma = -7,
		TokenAdd = -6,
		TokenMul = -5,
		TokenLeftBracket = -4,
		TokenRightBracket = -3,
	};

	// 词语Token
	struct Token
	{
		TokenType token_type = TokenType::TokenUnknown;
		int32_t start_pos = 0;	// 词语开始的位置
		int32_t end_pos = 0;	// 词语结束的位置
		Token(TokenType token_type, int32_t start_pos, int32_t end_pos): token_type(token_type), start_pos(start_pos), end_pos(end_pos) { }
	};

	// 语法树的节点
	struct TokenNode 
	{
		int32_t num_index = -1;
		std::shared_ptr<TokenNode> left = nullptr;	// 语法树的左节点
		std::shared_ptr<TokenNode> right = nullptr; // 语法树的右节点
		TokenNode(int32_t num_index, std::shared_ptr<TokenNode> left, std::shared_ptr<TokenNode> right);
		TokenNode() = default;
	};

	class ExpressionParser 
	{
	public:
		explicit ExpressionParser(std::string statement) : statement_(std::move(statement)) { }

		/**
		 * 词法分析
		 * @param re_tokenize 是否需要重新进行语法分析
		 */
		void Tokenizer(bool re_tokenize = false);

		/**
		 * 语法分析
		 * @return 生成的语法树
		 */
		std::vector<std::shared_ptr<TokenNode>> Generate();

		/**
		 * 返回词法分析的结果
		 * @return 词法分析的结果
		 */
		const std::vector<Token>& tokens() const;

		/**
		 * 返回词语字符串
		 * @return 词语字符串
		 */
		const std::vector<std::string>& token_strs() const;

	private:
		std::shared_ptr<TokenNode> Generate_(int32_t& index);
		std::vector<Token> tokens_;
		std::vector<std::string> token_strs_;
		std::string statement_;
	};
}
