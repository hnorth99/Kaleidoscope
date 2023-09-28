#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace llvm;

/////////////////////////////////////
/// Lexer
/////////////////////////////////////

// The lexer will parse through characters and returned tokenized interpretations of
// an inputted file. 
// The lexer passes over each character in the file and returns:
//  An enum mapping to defined tokens
//  A mapping to an ascii value (range 0-255) for undefined tokens

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

// The implementation of a lexer is this single function which 
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
  if (isdigit(last_char) || last_char == '.') { // Number: [0-9.]+
    std::string num_str;
    do {
      num_str += last_char;
      last_char = getchar();
    } while (isdigit(last_char) || last_char == '.');

    num_val = strtod(num_str.c_str(), nullptr);
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

/////////////////////////////////////
/// Abstract Syntax Tree
/////////////////////////////////////

namespace {

// Base class for all expression nodes.
class ExprAST {
  public:
    virtual ~ExprAST() = default;
    virtual Value *codegen() = 0;
};

// Expression class for numeric literals.
class NumberExprAST: public ExprAST {
  double val_;

  public:
    NumberExprAST(double val): val_(val) {}
    Value *codegen() override;
};

// Expression class for referencing variables.
class VariableExprAST : public ExprAST {
  std::string name_;

  public:
    VariableExprAST(const std::string &name): name_(name) {}
    Value *codegen() override;
};

// Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char op_;
  std::unique_ptr<ExprAST> lhs_, rhs_;
  
  public:
    BinaryExprAST(char op, std::unique_ptr<ExprAST> lhs,
                  std::unique_ptr<ExprAST> rhs)
      : op_(op), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}
    Value *codegen() override;
};

// Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string callee_;
  std::vector<std::unique_ptr<ExprAST>> args_;

  public:
    CallExprAST(const std::string &callee,
                std::vector<std::unique_ptr<ExprAST>> args)
      : callee_(callee), args_(std::move(args)) {}
    Value *codegen() override;
};

// This class represents the "prototype" for a function (name and args)
class PrototypeAST {
  std::string name_;
  std::vector<std::string> args_;

  public:
    PrototypeAST(const std::string &name, std::vector<std::string> args)
      : name_(name), args_(std::move(args)) {}

    const std::string &get_name() const { return name_; }
    Function *codegen();
};

// This class represents a function definition.
class FunctionAST {
  std::unique_ptr<PrototypeAST> proto_;
  std::unique_ptr<ExprAST> body_;

  public:
    FunctionAST(std::unique_ptr<PrototypeAST> proto,
                std::unique_ptr<ExprAST> body)
      : proto_(std::move(proto)), body_(std::move(body)) {}
    Function *codegen();
};

} // end anonymous namespace


/////////////////////////////////////
/// Parser
/////////////////////////////////////
// Provide a simple token buffer.
//  cur_tok is the current token the parser is looking at.  
//  getNextToken reads another token from the lexer and 
//  updates CurTok with its results.
static int cur_tok;
static int get_next_token() {
  return cur_tok = get_tok();
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

static std::unique_ptr<ExprAST> parse_expression();

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

// expression
//   ::= primary binoprhs
//
static std::unique_ptr<ExprAST> parse_expression() {
  auto lhs = parse_primary();
  if (!lhs)
    return nullptr;

  return parse_bin_ops_rhs(0, std::move(lhs));
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

/////////////////////////////////////
/// Code Gen
/////////////////////////////////////
// the_context objects owns lots of core llvm data structures
static std::unique_ptr<LLVMContext> the_context;
// the_module is an llvm construct that contains functions and
// global variables (owns the memory for all the generated IR)
static std::unique_ptr<Module> the_module;
// builder helps generate llvm instructions
static std::unique_ptr<IRBuilder<>> builder;
// named_values keeps track of which values are defined in the
// current scope and what their llvm representation is
static std::map<std::string, Value *> named_values;

Value *log_error_v(const char *str) {
  log_error(str);
  return nullptr;
}

Value *NumberExprAST::codegen() {
  return ConstantFP::get(*the_context, APFloat(val_));
}

Value *VariableExprAST::codegen() {
  // Look this variable up in the function.
  Value *v = named_values[name_];
  if (!v)
    log_error_v("Unknown variable name");
  return v;
}

Value *BinaryExprAST::codegen() {
  Value *l = lhs_->codegen();
  Value *r = rhs_->codegen();
  if (!l || !r)
    return nullptr;

  switch (op_) {
    case '+':
      return builder->CreateFAdd(l, r, "addtmp");
    case '-':
      return builder->CreateFSub(l, r, "subtmp");
    case '*':
      return builder->CreateFMul(l, r, "multmp");
    case '<':
      // Return a one bit integer
      l = builder->CreateFCmpULT(l, r, "cmptmp");
      // Convert bool 0/1 to double 0.0 or 1.0
      return builder->CreateUIToFP(l, Type::getDoubleTy(*the_context),
                                  "booltmp");
    default:
      return log_error_v("invalid binary operator");
  }
}

Value *CallExprAST::codegen() {
  // Look up the name in the global module table.
  Function *callee_f = the_module->getFunction(callee_);
  if (!callee_f)
    return log_error_v("Unknown function referenced");

  // If argument mismatch error.
  if (callee_f->arg_size() != args_.size())
    return log_error_v("Incorrect # arguments passed");

  // Codegen for all the arguments in the call
  std::vector<Value *> args_v;
  for (unsigned i = 0, e = args_.size(); i != e; ++i) {
    args_v.push_back(args_[i]->codegen());
    if (!args_v.back())
      return nullptr;
  }
  return builder->CreateCall(callee_f, args_v, "calltmp");
}

Function *PrototypeAST::codegen() {
  // Make a list of double types to match the arguments to function
  // prototype
  std::vector<Type*> doubles(args_.size(),
                             Type::getDoubleTy(*the_context));
  
  // Specify the arguments should be all double type, the function returns
  // a doubles type, and that the function is not varang
  FunctionType *ft =
    FunctionType::get(Type::getDoubleTy(*the_context), doubles, false);

  // Create the IR for the function prototype and which module to link it 
  // into (under the id name_)
  Function *f =
    Function::Create(ft, Function::ExternalLinkage, name_, the_module.get());

  // Set names for all arguments in the function.
  unsigned i = 0;
  for (auto &arg : f->args())
    arg.setName(args_[i++]);

  return f;
}

Function *FunctionAST::codegen() {
    // First, check for an existing function from a previous 'extern' declaration.
  Function *the_function = the_module->getFunction(proto_->get_name());

  // Confirm the function hasn't already been created (would be for externs)
  if (!the_function)
    the_function = proto_->codegen();

  if (!the_function)
    return nullptr;
  if (!the_function->empty())
    return (Function*)log_error_v("Function cannot be redefined.");

  // Create a new basic block to start insertion into.
  BasicBlock *bb = BasicBlock::Create(*the_context, "entry", the_function);
  builder->SetInsertPoint(bb);

  // Record the function arguments in the NamedValues map.
  named_values.clear();
  for (auto &arg : the_function->args())
    named_values[std::string(arg.getName())] = &arg;

  if (Value *ret_val = body_->codegen()) {
    // Finish off the function (return code generated by expression).
    builder->CreateRet(ret_val);
    // Validate the generated code, checking for consistency.
    verifyFunction(*the_function);
    return the_function;
  }

  // Error reading body, remove function.
  the_function->eraseFromParent();
  return nullptr;
}


// TODO: There is a bug because I am not validating the signature of a function against
// it's definition's own protoype. So, earlier extern declerations take precedence over
// function definitions:
// extern foo(a);     # ok, defines foo.
// def foo(b) b;      # Error: Unknown variable name. (decl using 'a' takes precedence).

/////////////////////////////////////
/// Handlers (top-level parsers) and JIT Driver
/////////////////////////////////////

static void initialize_module() {
  the_context = std::make_unique<LLVMContext>();
  the_module = std::make_unique<Module>("my jit", *the_context);
  builder = std::make_unique<IRBuilder<>>(*the_context);
}

static void handle_defintion() {
  if (auto fn_ast = parse_definition()) {
    if (auto *fn_ir = fn_ast->codegen()) {
      fprintf(stderr, "Parsed a function defintion:\n");
      fn_ir->print(errs());
      fprintf(stderr, "\n");
    }
  } else {
    // skip the token for error recovery.
    get_next_token();
  }
}

static void handle_extern() {
  if (auto proto_ast = parse_extern()) {
    if (auto *fn_ir = proto_ast->codegen()) {
      fprintf(stderr, "Read extern:\n");
      fn_ir->print(errs());
      fprintf(stderr, "\n");
    }
  } else {
    // skip the token for error recovery.
    get_next_token();
  }
}

static void handle_top_level_expression() {
  // Evaluate a top-level expression into an anonymous function.
  if (auto fn_ast = parse_top_level_expr()) {
    if (auto *fn_ir = fn_ast->codegen()) {
      fprintf(stderr, "Read top-level expression:\n");
      fn_ir->print(errs());
      fprintf(stderr, "\n");

      // Remove the anonymous expression.
      fn_ir->eraseFromParent();
    }
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

/////////////////////////////////////
/// Main / Driver code
/////////////////////////////////////
int main() {
  binop_precedence['<'] = 10;
  binop_precedence['+'] = 20;
  binop_precedence['-'] = 30;
  binop_precedence['*'] = 40; // highest precedence

  // Prime the first token.
  fprintf(stderr, "ready> ");
  get_next_token();

  // Make the module, which holds all the code.
  initialize_module();

  // Run the main interpreter loop.
  main_loop();

  // Print out all of the generated code.
  the_module->print(errs(), nullptr);

  return 0;
}
