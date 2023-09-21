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

static std::string IdentifierStr; // Filled in if tok_identifier
static double NumVal;             // Filled in if tok_number

// The implementation of a function is this single funciton which 
// is repeatedly called to return the next token from standard input
static int gettok() {
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
    
    // Return the identifier token
    return tok_identifier;
  }

  // Search for tok_number
  if (isdigit(LastChar || LastChar == '.')) { // Number: [0-9.]+
    std::string NumStr;
    do {
      NumStr += LastChar;
      LastChar = getchar();
    } while (isdigit(LastChar) || LastChar == '.');

    NumVal = strtod(NumStr.c_str(), 0);
    return tok_number;
  }  // TODO: Error handling (currently, 1.23.45 will be accepted as 1.23)

  // Skip comments (syntax: everything ignored after # until EOL)
  if (LastChar == '#') {
    do
      LastChar = getchar();
    while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');
    
    if (LastChar != EOF)
      return gettok()
  }

  // Search for EOF
  if (LastChar == EOF)
    return tok_eof;

  // Return non-reserved character as ascii value
  int ThisChar = LastChar;
  LastChar = getchar();
  return ThisChar;
}