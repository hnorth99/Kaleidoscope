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
static int gettok() {
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
      return gettok();
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
