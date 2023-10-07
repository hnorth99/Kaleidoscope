#include "include/KaleidoscopeJIT.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::orc;

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

  // control flow
  tok_if = -6,
  tok_then = -7,
  tok_else = -8,
  tok_for = -9, 
  tok_in = -10
};

static std::string IdentifierStr; // Filled in if tok_identifier
static double NumVal;             // Filled in if tok_number

// The implementation of a lexer is this single function which 
// is repeatedly called to return the next token from standard input
static int GetTok() {
  static int LastChar = ' ';

  // Skip whitespace
  while (isspace(LastChar))
    LastChar = getchar();

  // Search for tok_def, tok_extern, and tok_identifier
  if (isalpha(LastChar)) { // [a-zA-Z][a-zA-Z0-9]*
    // Build up the identifier
    IdentifierStr = LastChar;
    while (isalnum((LastChar = getchar())))
      IdentifierStr += LastChar;

    // Confirm that the identifier is not one of the other Tokens
    if (IdentifierStr == "def")
      return tok_def;
    if (IdentifierStr == "extern")
      return tok_extern;
    if (IdentifierStr == "if")
      return tok_if;
    if (IdentifierStr == "then")
      return tok_then;
    if (IdentifierStr == "else")
      return tok_else;
    if (IdentifierStr == "for")
      return tok_for;
    if (IdentifierStr == "in")
      return tok_in;
    
    // Return the identifier token
    return tok_identifier;
  }

  // Search for tok_number
  if (isdigit(LastChar) || LastChar == '.') { // Number: [0-9.]+
    std::string NumStr;
    do {
      NumStr += LastChar;
      LastChar = getchar();
    } while (isdigit(LastChar) || LastChar == '.');

    NumVal = strtod(NumStr.c_str(), nullptr);
    return tok_number;
  }  // TODO: Error handling (currently, 1.23.45 will be accepted as 1.23)

  // Skip comments (syntax: everything ignored after # until EOL)
  if (LastChar == '#') {
    do
      LastChar = getchar();
    while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');
    
    if (LastChar != EOF)
      return GetTok();
  }

  // Search for EOF
  if (LastChar == EOF)
    return tok_eof;

  // Return non-reserved character as ascii value
  int ThisChar = LastChar;
  LastChar = getchar();
  return ThisChar;
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
  double Val;

  public:
    NumberExprAST(double Val): Val(Val) {}
    Value *codegen() override;
};

// Expression class for referencing variables.
class VariableExprAST : public ExprAST {
  std::string Name;

  public:
    VariableExprAST(const std::string &Name): Name(Name) {}
    Value *codegen() override;
};

// Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;
  
  public:
    BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                  std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
    Value *codegen() override;
};

// Expression class for an if else workflow.
class IfExprAST : public ExprAST {
  std::unique_ptr<ExprAST> Cond, Then, Else;
  
  public:
    IfExprAST(std::unique_ptr<ExprAST> Cond,
              std::unique_ptr<ExprAST> Then,
              std::unique_ptr<ExprAST> Else)
      : Cond(std::move(Cond)), Then(std::move(Then)), Else(std::move(Else)) {}
    Value *codegen() override;
};

/// ForExprAST - Expression class for for/in.
class ForExprAST : public ExprAST {
  std::string VarName;
  std::unique_ptr<ExprAST> Start, End, Step, Body;

  public:
    ForExprAST(const std::string &VarName, std::unique_ptr<ExprAST> Start,
              std::unique_ptr<ExprAST> End, std::unique_ptr<ExprAST> Step,
              std::unique_ptr<ExprAST> Body)
      : VarName(VarName), Start(std::move(Start)), End(std::move(End)),
        Step(std::move(Step)), Body(std::move(Body)) {}

    Value *codegen() override;
};

// Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string Callee;
  std::vector<std::unique_ptr<ExprAST>> Args;

  public:
    CallExprAST(const std::string &Callee,
                std::vector<std::unique_ptr<ExprAST>> Args)
      : Callee(Callee), Args(std::move(Args)) {}
    Value *codegen() override;
};

// This class represents the "prototype" for a function (name and args)
class PrototypeAST {
  std::string Name;
  std::vector<std::string> Args;

  public:
    PrototypeAST(const std::string &Name, std::vector<std::string> Args)
      : Name(Name), Args(std::move(Args)) {}

    const std::string &getName() const { return Name; }
    Function *codegen();
};

// This class represents a function definition.
class FunctionAST {
  std::unique_ptr<PrototypeAST> Proto;
  std::unique_ptr<ExprAST> Body;

  public:
    FunctionAST(std::unique_ptr<PrototypeAST> Proto,
                std::unique_ptr<ExprAST> Body)
      : Proto(std::move(Proto)), Body(std::move(Body)) {}
    Function *codegen();
};

} // end anonymous namespace


/////////////////////////////////////
/// Parser
/////////////////////////////////////
// Provide a simple token buffer.
//  CurTok is the current token the parser is looking at.  
//  getNextToken reads another token from the lexer and 
//  updates CurTok with its results.
static int CurTok;
static int getNextToken() {
  return CurTok = GetTok();
}
// BinopPrecedence - This holds the precedence for each binary operator that is
// defined.
// TODO add more binary operations
static std::map<char, int> BinopPrecedence;

// GetTok_precedence - Get the precedence of the pending binary operator token.
static int getTokPrecedence() {
  if (!isascii(CurTok))
    return -1;

  int TokPrec = BinopPrecedence[CurTok];
  if (TokPrec <= 0) return -1;
  return TokPrec;
}

// These are helper functions for error handling.
// TODO: Come up with better error handling routine
std::unique_ptr<ExprAST> logError(const char *Str) { // TODO: why char*?
  fprintf(stderr, "Error: %s\n", Str);
  return nullptr;
}

std::unique_ptr<PrototypeAST> logErrorP(const char *Str) {
  logError(Str);
  return nullptr;
}

static std::unique_ptr<ExprAST> parseExpression();

// numberexpr ::= number
static std::unique_ptr<ExprAST> parseNumberExpr() {
  auto Result = std::make_unique<NumberExprAST>(NumVal);
  getNextToken();
  return std::move(Result);
}

// parenexpr ::= '(' expression ')'
static std::unique_ptr<ExprAST> parseParenExpr() {
  getNextToken(); // consume a (.
  auto V = parseExpression();
  if (!V)
    return nullptr;

  if (CurTok != ')')
    return logError("expected ')'");
  getNextToken(); // consume ).
  return V;
}

// identifierexpr
//   ::= identifier
//   ::= identifier '(' expression* ')'
static std::unique_ptr<ExprAST> parseIdentifierExpr() {
  std::string IdName = IdentifierStr;
  getNextToken();  // consume identifier.

  // Simple variable ref.
  if (CurTok != '(')
    return std::make_unique<VariableExprAST>(IdName);

  // Call, need to build up args.
  getNextToken();  // consume (.
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while (true) {
      if (auto Arg = parseExpression())
        Args.push_back(std::move(Arg));
      else
        return nullptr;

      if (CurTok == ')')
        break;

      if (CurTok != ',')
        return logError("Expected ')' or ',' in argument list");
      getNextToken();
    }
  }
  // Consume the ')'.
  getNextToken();
  return std::make_unique<CallExprAST>(IdName, std::move(Args));
}

// ifexpr ::= 'if' expression 'then' expression 'else' expression
static std::unique_ptr<ExprAST> parseIfExpr() {
  getNextToken();  // consume if.

  auto Cond = parseExpression();
  if (!Cond)
    return nullptr;

  if (CurTok != tok_then)
    return logError("expected then");
  getNextToken(); // consume then.

  auto Then = parseExpression();
  if (!Then)
    return nullptr;

  if (CurTok != tok_else)
    return logError("expected else");
  getNextToken(); // consume else.

  auto Else = parseExpression();
  if (!Else)
    return nullptr;
  
  return std::make_unique<IfExprAST>(std::move(Cond), std::move(Then), 
                                      std::move(Else));
}

// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
static std::unique_ptr<ExprAST> parseForExpr() {
  getNextToken();  // consume for.

  if (CurTok != tok_identifier)
    return logError("expected identifier after for");
  std::string IdName = IdentifierStr;
  getNextToken();  // consume identifier.

  if (CurTok != '=')
    return logError("expected '=' after for");
  getNextToken();  // consume '='.

  auto Start = parseExpression();
  if (!Start)
    return nullptr;
  if (CurTok != ',')
    return logError("expected ',' after for start value");
  getNextToken();

  auto End = parseExpression();
  if (!End)
    return nullptr;

  // The step value is optional.
  std::unique_ptr<ExprAST> Step;
  if (CurTok == ',') {
    getNextToken();
    Step = parseExpression();
    if (!Step)
      return nullptr;
  }

  if (CurTok != tok_in)
    return logError("expected 'in' after for");
  getNextToken();  // consume 'in'.

  auto Body = parseExpression();
  if (!Body)
    return nullptr;

  return std::make_unique<ForExprAST>(IdName, std::move(Start),
                                       std::move(End), std::move(Step),
                                       std::move(Body));
}

// primary
//   ::= identifierexpr
//   ::= numberexpr
//   ::= parenexpr
static std::unique_ptr<ExprAST> parsePrimary() {
  switch (CurTok) {
    case tok_identifier:
      return parseIdentifierExpr();
    case tok_number:
      return parseNumberExpr();
    case tok_if:
      return parseIfExpr();
    case tok_for:
      return parseForExpr();
    case '(':
      return parseParenExpr();
    default:
      return logError("unknown token when expecting an expression");
  }
}

// binoprhs
//   ::= ('+' primary)*
static std::unique_ptr<ExprAST> parseBinOpsRHS(int ExprPrec,
                                              std::unique_ptr<ExprAST> LHS) {
  // If this is a binop, find its precedence.
  while (true) {
    int TokPrec = getTokPrecedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    if (TokPrec < ExprPrec)
      return LHS;

    // Okay, we know this is a binop.
    int BinOp = CurTok;
    getNextToken();  // consume binop

    // Parse the primary expression after the binary operator.
    auto RHS = parsePrimary();
    if (!RHS)
      return nullptr;
    
    // If bin_op binds less tightly with rhs than the operator after rhs, let
    // the pending operator take rhs as its LHS.
    int NextPrec = getTokPrecedence();
    if (TokPrec < NextPrec) {
      RHS = parseBinOpsRHS(TokPrec+1, std::move(RHS));
      if (!RHS)
        return nullptr;
    }

    // Merge LHS/rhs.
    LHS = std::make_unique<BinaryExprAST>(BinOp, std::move(LHS),
                                           std::move(RHS));
  }
}

// expression
//   ::= primary binoprhs
//
static std::unique_ptr<ExprAST> parseExpression() {
  auto LHS = parsePrimary();
  if (!LHS)
    return nullptr;

  return parseBinOpsRHS(0, std::move(LHS));
}

// prototype
//   ::= id '(' id* ')'
static std::unique_ptr<PrototypeAST> parsePrototype() {
  if (CurTok != tok_identifier)
    return logErrorP("Expected function name in protoype");

  std::string FnName = IdentifierStr;
  getNextToken();

  if (CurTok != '(')
    return logErrorP("Expected '(' to be in prototype");

  // Read the list of argument names.
  std::vector<std::string> ArgNames;
  while(getNextToken() == tok_identifier)
    ArgNames.push_back(IdentifierStr);

  if (CurTok != ')')
    return logErrorP("Expected ')' to be in prototype");

  // success.
  getNextToken();  // consume ')'.

  return std::make_unique<PrototypeAST>(FnName, std::move(ArgNames));
}

// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> parseDefinition() {
  getNextToken(); // consume def.
  auto Proto = parsePrototype();
  if (!Proto) return nullptr;

  if (auto E = parseExpression())
    return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));

  return nullptr;
}

// external ::= 'extern' prototype
static std::unique_ptr<PrototypeAST> parseExtern() {
  getNextToken(); // consume extern.
  return parsePrototype();
}

// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> parseTopLevelExpr() {
  if (auto E = parseExpression()) {
    auto Proto = std::make_unique<PrototypeAST>("__anon_expr", std::vector<std::string>());
    return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  }
  return nullptr;
}

/////////////////////////////////////
/// Code Gen
/////////////////////////////////////
// TheContext objects owns lots of core llvm data structures
static std::unique_ptr<LLVMContext> TheContext;
// TheModule is an llvm construct that contains functions and
// global variables (owns the memory for all the generated IR)
static std::unique_ptr<Module> TheModule;
// TheFPM providers an interface to add optimizations
static std::unique_ptr<legacy::FunctionPassManager> TheFPM;
static std::unique_ptr<KaleidoscopeJIT> TheJIT;
static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;
// Builder helps generate llvm instructions
static std::unique_ptr<IRBuilder<>> Builder;
// NamedValues keeps track of which values are defined in the
// current scope and what their llvm representation is
static std::map<std::string, Value *> NamedValues;
static ExitOnError ExitOnErr;

Value *logErrorV(const char *Str) {
  logError(Str);
  return nullptr;
}

Value *NumberExprAST::codegen() {
  return ConstantFP::get(*TheContext, APFloat(Val));
}

Value *VariableExprAST::codegen() {
  // Look this variable up in the function.
  Value *V = NamedValues[Name];
  if (!V)
    logErrorV("Unknown variable name");
  return V;
}

Value *BinaryExprAST::codegen() {
  Value *L = LHS->codegen();
  Value *R = RHS->codegen();
  if (!L || !R)
    return nullptr;

  switch (Op) {
    case '+':
      return Builder->CreateFAdd(L, R, "addtmp");
    case '-':
      return Builder->CreateFSub(L, R, "subtmp");
    case '*':
      return Builder->CreateFMul(L, R, "multmp");
    case '<':
      // Return a one bit integer
      L = Builder->CreateFCmpULT(L, R, "cmptmp");
      // Convert bool 0/1 to double 0.0 or 1.0
      return Builder->CreateUIToFP(L, Type::getDoubleTy(*TheContext),
                                  "booltmp");
    default:
      return logErrorV("invalid binary operator");
  }
}

Function *getFunction(std::string Name) {
  // First, see if the function has already been added to the current module.
  if (auto *F = TheModule->getFunction(Name))
    return F;

  // If not, check whether we can codegen the declaration from some existing
  // prototype.
  auto FI = FunctionProtos.find(Name);
  if (FI != FunctionProtos.end())
    return FI->second->codegen();

  // If no existing prototype exists, return null.
  return nullptr;
}

Value *CallExprAST::codegen() {
  // Look up the name in the global module table.
  Function *CalleeF = getFunction(Callee);
  if (!CalleeF)
    return logErrorV("Unknown function referenced");

  // If argument mismatch error.
  if (CalleeF->arg_size() != Args.size())
    return logErrorV("Incorrect # arguments passed");

  // Codegen for all the arguments in the call
  std::vector<Value *> ArgsV;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    ArgsV.push_back(Args[i]->codegen());
    if (!ArgsV.back())
      return nullptr;
  }
  return Builder->CreateCall(CalleeF, ArgsV, "calltmp");
}

Value *IfExprAST::codegen() {
  Value *CondV = Cond->codegen();
  if (!CondV)
    return nullptr;

  // Convert condition to a bool
  CondV = Builder->CreateFCmpONE(
    CondV, ConstantFP::get(*TheContext, APFloat(0.0)), "ifcond");

  // Ask the Builder for the parent of the current block 
  Function *TheFunction = Builder->GetInsertBlock()->getParent();

  // Start inserting block for if and else cases... follow with then block
  BasicBlock *ThenBB = BasicBlock::Create(*TheContext, "then", TheFunction);
  BasicBlock *ElseBB = BasicBlock::Create(*TheContext, "else");
  BasicBlock *MergeBB = BasicBlock::Create(*TheContext, "ifcont");
  Builder->CreateCondBr(CondV, ThenBB, ElseBB);

  // Emit then value
  Builder->SetInsertPoint(ThenBB);
  Value *ThenV = Then->codegen();
  if (!ThenV)
    return nullptr;
  Builder->CreateBr(MergeBB);
  // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
  ThenBB = Builder->GetInsertBlock();

  // Emit then value
  TheFunction->insert(TheFunction->end(), ElseBB);
  Builder->SetInsertPoint(ElseBB);
  Value *ElseV = Else->codegen();
  if (!ElseV)
    return nullptr;
  Builder->CreateBr(MergeBB);
  // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
  ElseBB = Builder->GetInsertBlock();

  // Emit merge block.
  TheFunction->insert(TheFunction->end(), MergeBB);
  Builder->SetInsertPoint(MergeBB);
  PHINode *PN =
    Builder->CreatePHI(Type::getDoubleTy(*TheContext), 2, "iftmp");
  PN->addIncoming(ThenV, ThenBB);
  PN->addIncoming(ElseV, ElseBB);

  return PN;
}

Value *ForExprAST::codegen() {
  // Emit the start code first, without 'variable' in scope.
  Value *StartVal = Start->codegen();
  if (!StartVal)
    return nullptr;

  // Make the new basic block for the loop header, inserting after current
  // block.
  Function *TheFunction = Builder->GetInsertBlock()->getParent();
  BasicBlock *PreHeaderBB = Builder->GetInsertBlock();
  BasicBlock *LoopBB =
      BasicBlock::Create(*TheContext, "loop", TheFunction);

  // Insert an explicit fall through from the current block to the LoopBB.
  Builder->CreateBr(LoopBB);

  // Start insertion in LoopBB.
  Builder->SetInsertPoint(LoopBB);

  // Start the PHI node with an entry for Start.
  PHINode *Variable = Builder->CreatePHI(Type::getDoubleTy(*TheContext),
                                        2, VarName);
  Variable->addIncoming(StartVal, PreHeaderBB);

  // Within the loop, the variable is defined equal to the PHI node.  If it
  // shadows an existing variable, we have to restore it, so save it now.
  Value *OldVal = NamedValues[VarName];
  NamedValues[VarName] = Variable;

  // Emit the body of the loop.  This, like any other expr, can change the
  // current BB.  Note that we ignore the value computed by the body, but don't
  // allow an error.
  if (!Body->codegen())
    return nullptr;

  // Emit the step value.
  Value *StepVal = nullptr;
  if (Step) {
    StepVal = Step->codegen();
    if (!StepVal)
      return nullptr;
  } else {
    // If not specified, use 1.0.
    StepVal = ConstantFP::get(*TheContext, APFloat(1.0));
  }
  Value *NextVar = Builder->CreateFAdd(Variable, StepVal, "nextvar");

  // Compute the end condition.
  Value *EndCond = End->codegen();
  if (!EndCond)
    return nullptr;

  // Convert condition to a bool by comparing non-equal to 0.0.
  EndCond = Builder->CreateFCmpONE(
      EndCond, ConstantFP::get(*TheContext, APFloat(0.0)), "loopcond");

  // Create the "after loop" block and insert it.
  BasicBlock *LoopEndBB = Builder->GetInsertBlock();
  BasicBlock *AfterBB =
      BasicBlock::Create(*TheContext, "afterloop", TheFunction);
  // Insert the conditional branch into the end of loop_end_bb.
  Builder->CreateCondBr(EndCond, LoopBB, AfterBB);

  // Any new code will be inserted in AfterBB.
  Builder->SetInsertPoint(AfterBB);

  // Add a new entry to the PHI node for the backedge.
  Variable->addIncoming(NextVar, LoopEndBB);

  // Restore the unshadowed variable.
  if (OldVal)
    NamedValues[VarName] = OldVal;
  else
    NamedValues.erase(VarName);
  // for expr always returns 0.0.
  return Constant::getNullValue(Type::getDoubleTy(*TheContext));
}

Function *PrototypeAST::codegen() {
  // Make a list of double types to match the arguments to function
  // prototype
  std::vector<Type*> Doubles(Args.size(),
                             Type::getDoubleTy(*TheContext));
  
  // Specify the arguments should be all double type, the function returns
  // a doubles type, and that the function is not varang
  FunctionType *FT =
    FunctionType::get(Type::getDoubleTy(*TheContext), Doubles, false);

  // Create the IR for the function prototype and which module to link it 
  // into (under the id name_)
  Function *F =
    Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

  // Set names for all arguments in the function.
  unsigned i = 0;
  for (auto &Arg : F->args())
    Arg.setName(Args[i++]);

  return F;
}

Function *FunctionAST::codegen() {
  // Transfer ownership of the prototype to the FunctionProtos map, but keep a
  // reference to it for use below.
  auto &P = *Proto;
  FunctionProtos[Proto->getName()] = std::move(Proto); // TODO: understand why this is safe 
  Function *TheFunction = getFunction(P.getName());
  if (!TheFunction)
    return nullptr;

  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(*TheContext, "entry", TheFunction);
  Builder->SetInsertPoint(BB);

  // Record the function arguments in the NamedValues map.
  NamedValues.clear();
  for (auto &Arg : TheFunction->args())
    NamedValues[std::string(Arg.getName())] = &Arg;

  if (Value *RetVal = Body->codegen()) {
    // Finish off the function (return code generated by expression).
    Builder->CreateRet(RetVal);
    // Validate the generated code, checking for consistency.
    verifyFunction(*TheFunction);
    // Optimize function
    TheFPM->run(*TheFunction);
    return TheFunction;
  }

  // Error reading body, remove function.
  TheFunction->eraseFromParent();
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

void initializeModuleAndPassManager(void) {
  TheContext = std::make_unique<LLVMContext>();
  TheModule = std::make_unique<Module>("my cool jit", *TheContext);
  TheModule->setDataLayout(TheJIT->getDataLayout());
  Builder = std::make_unique<IRBuilder<>>(*TheContext);

  // Create a new pass manager attached to it.
  TheFPM = std::make_unique<legacy::FunctionPassManager>(TheModule.get());

  // Do simple "peephole" optimizations and bit-twiddling optzns.
  TheFPM->add(createInstructionCombiningPass());
  // Reassociate expressions.
  TheFPM->add(createReassociatePass());
  // Eliminate Common SubExpressions.
  TheFPM->add(createGVNPass());
  // Simplify the control flow graph (deleting unreachable blocks, etc).
  TheFPM->add(createCFGSimplificationPass());

  TheFPM->doInitialization();
}

static void handleDefintion() {
  if (auto FnAST = parseDefinition()) {
    if (auto *FnIR = FnAST->codegen()) {
      fprintf(stderr, "Parsed a function defintion:\n");
      FnIR->print(errs());
      fprintf(stderr, "\n");
      ExitOnErr(TheJIT->addModule(
          ThreadSafeModule(std::move(TheModule), std::move(TheContext))));
      initializeModuleAndPassManager();
    }
  } else {
    // skip the token for error recovery.
    getNextToken();
  }
}

static void handleExtern() {
  if (auto ProtoAST = parseExtern()) {
    if (auto *FnIR = ProtoAST->codegen()) {
      fprintf(stderr, "Read extern:\n");
      FnIR->print(errs());
      fprintf(stderr, "\n");
      FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
    }
  } else {
    // skip the token for error recovery.
    getNextToken();
  }
}

static void handleTopLevelExpression() {
  // Evaluate a top-level expression into an anonymous function.
  if (auto FnAST = parseTopLevelExpr()) {
    if (FnAST->codegen()) {
      // Create a ResourceTracker to track JIT'd memory allocated to our
      // anonymous expression -- that way we can free it after executing.
      auto RT = TheJIT->getMainJITDylib().createResourceTracker();

      auto TSM = ThreadSafeModule(std::move(TheModule), std::move(TheContext));
      ExitOnErr(TheJIT->addModule(std::move(TSM), RT));
      // Once the module has been added to the JIT it can no longer be modified, so 
      // open a new module to hold subsequent code
      initializeModuleAndPassManager();

      // Search the JIT for the __anon_expr symbol (top level expression).
      auto ExprSymbol = ExitOnErr(TheJIT->lookup("__anon_expr"));

      // Get the symbol's address and cast it to the right type (takes no
      // arguments, returns a double) so we can call it as a native function.
      double (*fp)() = ExprSymbol.getAddress().toPtr<double (*)()>();
      fprintf(stderr, "Evaluated to %f\n", fp());

      // Delete the anonymous expression module from the JIT (since this doesn't 
      // support re-evaluation).
      ExitOnErr(RT->remove());
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

// top ::= definition | external | expression | ';'
static void mainLoop() {
  while (true) {
    fprintf(stderr, "ready> ");
    switch (CurTok) {
      case tok_eof:
        return;
      case ';': // ignore top-level semicolons;
        getNextToken();
        break;
      case tok_def:
        handleDefintion();
        break;
      case tok_extern:
        handleExtern();
        break;
      default:
        handleTopLevelExpression();
        break;
    }
  }
  
}

/////////////////////////////////////
/// Main / Driver code
/////////////////////////////////////
int main() {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 30;
  BinopPrecedence['*'] = 40; // highest precedence

  // Prime the first token.
  fprintf(stderr, "ready> ");
  getNextToken();

  TheJIT = ExitOnErr(KaleidoscopeJIT::Create());

  // Make the module, which holds all the code.
  initializeModuleAndPassManager();

  // Run the main interpreter loop.
  mainLoop();

  // Print out all of the generated code.
  TheModule->print(errs(), nullptr);

  return 0;
}
