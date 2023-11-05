#pragma once

#include <string>
#include <utility>
#include <vector>
#include <memory>

namespace my_infer 
{
	// ���������
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

	// ����Token
	struct Token
	{
		TokenType token_type = TokenType::TokenUnknown;
		int32_t start_pos = 0;	// ���￪ʼ��λ��
		int32_t end_pos = 0;	// ���������λ��
		Token(TokenType token_type, int32_t start_pos, int32_t end_pos): token_type(token_type), start_pos(start_pos), end_pos(end_pos) { }
	};

	// �﷨���Ľڵ�
	struct TokenNode 
	{
		int32_t num_index = -1;
		std::shared_ptr<TokenNode> left = nullptr;	// �﷨������ڵ�
		std::shared_ptr<TokenNode> right = nullptr; // �﷨�����ҽڵ�
		TokenNode(int32_t num_index, std::shared_ptr<TokenNode> left, std::shared_ptr<TokenNode> right);
		TokenNode() = default;
	};

	class ExpressionParser 
	{
	public:
		explicit ExpressionParser(std::string statement) : statement_(std::move(statement)) { }

		/**
		 * �ʷ�����
		 * @param re_tokenize �Ƿ���Ҫ���½����﷨����
		 */
		void Tokenizer(bool re_tokenize = false);

		/**
		 * �﷨����
		 * @return ���ɵ��﷨��
		 */
		std::vector<std::shared_ptr<TokenNode>> Generate();

		/**
		 * ���شʷ������Ľ��
		 * @return �ʷ������Ľ��
		 */
		const std::vector<Token>& tokens() const;

		/**
		 * ���ش����ַ���
		 * @return �����ַ���
		 */
		const std::vector<std::string>& token_strs() const;

	private:
		std::shared_ptr<TokenNode> Generate_(int32_t& index);
		std::vector<Token> tokens_;
		std::vector<std::string> token_strs_;
		std::string statement_;
	};
}
