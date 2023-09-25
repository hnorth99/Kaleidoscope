// The lexer will parse through characters and returned tokenized interpretations of
// an inputted file. 
// The lexer passes over each character in the file and returns:
//  An enum mapping to defined tokens
//  A mapping to an ascii value (range 0-255) for undefined tokens

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

enum Token {
  tok_eof = -1,

  // commands
  tok_def = -2,
  tok_extern = -3,

  // primary
  tok_identifier = -4,
  tok_number = -5,
};

static std::string identifier_str; // Filled in if tok_identifier
static double num_val;             // Filled in if tok_number

// The implementation of a function is this single function which 
// is repeatedly called to return the next token from standard input
static int get_tok() {
  static int last_char = ' ';

  // Skip whitespace
  while (isspace(last_char))
    last_char = getchar();

  // Search for tok_def, tok_extern, and tok_identifier
  if (isalpha(last_char)) { // [a-zA-Z][a-zA-Z0-9]*
    // Build up the identifier
    identifier_str = last_char;
    while (isalnum((last_char = getchar())))
      identifier_str += last_char;

    // Confirm that the identifier is not one of the other Tokens
    if (identifier_str == "def")
      return tok_def;
    if (identifier_str == "extern")
      return tok_extern;
    
    // Return the identifier token
    return tok_identifier;
  }

  // Search for tok_number
  if (isdigit(last_char || last_char == '.')) { // Number: [0-9.]+
    std::string NumStr;
    do {
      NumStr += last_char;
      last_char = getchar();
    } while (isdigit(last_char) || last_char == '.');

    num_val = strtod(NumStr.c_str(), 0);
    return tok_number;
  }  // TODO: Error handling (currently, 1.23.45 will be accepted as 1.23)

  // Skip comments (syntax: everything ignored after # until EOL)
  if (last_char == '#') {
    do
      last_char = getchar();
    while (last_char != EOF && last_char != '\n' && last_char != '\r');
    
    if (last_char != EOF)
      return get_tok();
  }

  // Search for EOF
  if (last_char == EOF)
    return tok_eof;

  // Return non-reserved character as ascii value
  int this_char = last_char;
  last_char = getchar();
  return this_char;
}

// Base class for all expression nodes.
class ExprAST {
  public:
    virtual ~ExprAST() = default;
};

// Expression class for numeric literals.
class NumberExprAST: public ExprAST {
  double val_;

  public:
    NumberExprAST(double val): val_(val) {}
};

// Expression class for referencing variables.
class VariableExprAST : public ExprAST {
  std:: string name_;

  public:
    VariableExprAST(const std::string &name): name_(name) {}
};

// Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char op_;
  std::unique_ptr<ExprAST> lhs_, rhs_;
  
  public:
    BinaryExprAST(char op, std::unique_ptr<ExprAST> lhs,
                  std::unique_ptr<ExprAST> rhs)
      : op_(op), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}
};

// Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string callee_;
  std::vector<std::unique_ptr<ExprAST>> args_;

  public:
    CallExprAST(const std::string &callee,
                std::vector<std::unique_ptr<ExprAST>> args)
      : callee_(callee), args_(std::move(args)) {}
};

// This class represents the "prototype" for a function (name and args)
class PrototypeAST {
  std::string name_;
  std::vector<std::string> args_;

public:
  PrototypeAST(const std::string &name, std::vector<std::string> args)
    : name_(name), args_(std::move(args)) {}

  const std::string &get_name() const { return name_; }
};

// This class represents a function definition.
class FunctionAST {
  std::unique_ptr<PrototypeAST> proto_;
  std::unique_ptr<ExprAST> body_;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> proto,
              std::unique_ptr<ExprAST> body)
    : proto_(std::move(proto)), body_(std::move(body)) {}
};

// These are helper functions for error handling.
// TODO: Come up with better error handling routine
std::unique_ptr<ExprAST> log_error(const char *str) { // TODO: why char*?
  fprintf(stderr, "Error: %s\n", str);
  return nullptr;
}

std::unique_ptr<PrototypeAST> log_error_p(const char *str) {
  log_error(str);
  return nullptr;
}

// Provide a simple token buffer.
//  cur_tok is the current token the parser is looking at.  
//  getNextToken reads another token from the lexer and 
//  updates CurTok with its results.
static int cur_tok;
static int get_next_token() {
  return cur_tok = get_tok();
}

// numberexpr ::= number
static std::unique_ptr<ExprAST> parse_number_expr() {
  auto result = std::make_unique<NumberExprAST>(num_val);
  get_next_token();
  return std::move(result);
}

// parenexpr ::= '(' expression ')'
static std::unique_ptr<ExprAST> parse_paren_expr() {
  get_next_token(); // consume a (.
  auto v = parse_expression();
  if (!v)
    return nullptr;

  if (cur_tok != ')')
    return log_error("expected ')'");
  get_next_token(); // consume ).
  return v;
}

// identifierexpr
//   ::= identifier
//   ::= identifier '(' expression* ')'
static std::unique_ptr<ExprAST> parse_identifier_expr() {
  std::string id_name = identifier_str;
  get_next_token();  // consume identifier.

  // Simple variable ref.
  if (cur_tok != '(')
    return std::make_unique<VariableExprAST>(id_name);

  // Call, need to build up args.
  get_next_token();  // consume (.
  std::vector<std::unique_ptr<ExprAST>> args;
  if (cur_tok != ')') {
    while (true) {
      if (auto arg = parse_expression())
        args.push_back(std::move(arg));
      else
        return nullptr;

      if (cur_tok == ')')
        break;

      if (cur_tok != ',')
        return log_error("Expected ')' or ',' in argument list");
      get_next_token();
    }
  }
  // Consume the ')'.
  get_next_token();
  return std::make_unique<CallExprAST>(id_name, std::move(args));
}

// primary
//   ::= identifierexpr
//   ::= numberexpr
//   ::= parenexpr
static std::unique_ptr<ExprAST> parse_primary() {
  switch (cur_tok) {
    case tok_identifier:
      return parse_identifier_expr();
    case tok_number:
      return parse_number_expr();
    case '(':
      return parse_paren_expr();
    default:
      return log_error("unknown token when expecting an expression");
  }
}

// BinopPrecedence - This holds the precedence for each binary operator that is
// defined.
// TODO add more binary operations
static std::map<char, int> binop_precedence;

// get_tok_precedence - Get the precedence of the pending binary operator token.
static int get_tok_precedence() {
  if (!isascii(cur_tok))
    return -1;

  int tok_prec = binop_precedence[cur_tok];
  if (tok_prec <= 0) return -1;
  return tok_prec;
}

int main() {
  binop_precedence['<'] = 10;
  binop_precedence['+'] = 20;
  binop_precedence['-'] = 30;
  binop_precedence['*'] = 40; // highest precedence
}

// expression
//   ::= primary binoprhs
//
static std::unique_ptr<ExprAST> parse_expression() {
  auto lhs = parse_primary();
  if (!lhs)
    return nullptr;

  return parse_bin_ops_rhs(0, std::move(lhs));
}

// binoprhs
//   ::= ('+' primary)*
static std::unique_ptr<ExprAST> parse_bin_ops_rhs(int expr_prec,
                                              std::unique_ptr<ExprAST> lhs) {
  // If this is a binop, find its precedence.
  while (true) {
    int tok_prec = get_tok_precedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    if (tok_prec < expr_prec)
      return lhs;

    // Okay, we know this is a binop.
    int bin_op = cur_tok;
    get_next_token();  // consume binop

    // Parse the primary expression after the binary operator.
    auto rhs = parse_primary();
    if (!rhs)
      return nullptr;
    
    // If bin_op binds less tightly with rhs than the operator after rhs, let
    // the pending operator take rhs as its lhs.
    int next_prec = get_tok_precedence();
    if (tok_prec < next_prec) {
      rhs = parse_bin_ops_rhs(tok_prec+1, std::move(rhs));
      if (!rhs)
        return nullptr;
    }

    // Merge lhs/rhs.
    lhs = std::make_unique<BinaryExprAST>(bin_op, std::move(lhs),
                                           std::move(rhs));
  }
}

// prototype
//   ::= id '(' id* ')'
static std::unique_ptr<PrototypeAST> parse_prototype() {
  if (cur_tok != tok_identifier)
    return log_error_p("Expected function name in protoype");

  std::string fn_name = identifier_str;
  get_next_token();

  if (cur_tok != '(')
    return log_error_p("Expected '(' to be in prototype");

  // Read the list of argument names.
  std::vector<std::string> arg_names;
  while(get_next_token() == tok_identifier)
    arg_names.push_back(identifier_str);

  if (cur_tok != ')')
    return log_error_p("Expected ')' to be in prototype");

  // success.
  get_next_token();  // consume ')'.

  return std::make_unique<PrototypeAST>(fn_name, std::move(arg_names));
}

// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> parse_definition() {
  get_next_token(); // consume def.
  auto proto = parse_prototype();
  if (!proto) return nullptr;

  if (auto e = parse_expression())
    return std::make_unique<FunctionAST>(std::move(proto), std::move(e));

  return nullptr;
}

// external ::= 'extern' prototype
static std::unique_ptr<PrototypeAST> parse_extern() {
  get_next_token(); // consume extern.
  return parse_prototype();
}

// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> parse_top_level_expr() {
  if (auto e = parse_expression()) {
    auto proto = std::make_unique<PrototypeAST>("", std::vector<std::string>());
    return std::make_unique<FunctionAST>(std::move(proto), std::move(e));
  }
  return nullptr;
}

static void handle_defintion() {
  if (parse_definition()) {
    fprintf(stderr, "Parsed a function defintion\n");
  } else {
    // skip the token for error recovery.
    get_next_token();
  }
}

static void handle_extern() {
  if (parse_extern()) {
    fprintf(stderr, "Parsed an extern\n");
  } else {
    // skip the token for error recovery.
    get_next_token();
  }
}

static void handle_top_level_expression() {
  // Evaluate a top-level expression into an anonymous function.
  if (parse_top_level_expr()) {
    fprintf(stderr, "Parsed a top-level expr\n");
  } else {
    // Skip token for error recovery.
    get_next_token();
  }
}

// top ::= definition | external | expression | ';'
static void main_loop() {
  while (true) {
    fprintf(stderr, "ready> ");
    switch (cur_tok) {
      case tok_eof:
        return;
      case ';': // ignore top-level semicolons;
        get_next_token();
        break;
      case tok_def:
        handle_defintion();
        break;
      case tok_extern:
        handle_extern();
        break;
      default:
        handle_top_level_expression();
        break;
    }
  }
}
