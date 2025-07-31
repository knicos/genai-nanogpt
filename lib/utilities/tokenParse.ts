function wordToken(text: string, tokens: string[]) {
    if (text.length === 0) {
        return;
    }
    tokens.push(` ${text.trim()}`);
}

export default function parseTokens(text: string, raw?: boolean): string[] {
    const normalizedText = !raw ? text.toLocaleLowerCase() : text;
    const tokens: string[] = [];

    let currentToken = '';
    for (let i = 0; i < normalizedText.length; i++) {
        const char = normalizedText[i];

        switch (char) {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
            case ':':
            case ';':
            case ',':
            case '.':
            case '?':
            case '!':
            case '"':
            case "'":
            case '`':
            case '(':
            case ')':
            case '[':
            case ']':
            case '{':
            case '}':
            case '-':
            case '_':
            case '/':
            case '\\':
            case '%':
            case '<':
            case '>':
            case '=':
            case '+':
            case '*':
            case '&':
            case '^':
            case '|':
            case '~':
            case '@':
            case '#':
            case '$':
                if (raw) tokens.push(currentToken);
                else wordToken(currentToken, tokens);
                tokens.push(char);
                currentToken = '';
                break;
            case ' ':
                if (raw) tokens.push(currentToken);
                else wordToken(currentToken, tokens);
                currentToken = char;
                break;
            default:
                currentToken += char;
                break;
        }
    }

    if (currentToken.length > 0) {
        // Add the last token if it exists
        if (raw) tokens.push(currentToken);
        else wordToken(currentToken, tokens);
    }

    return tokens;
}
