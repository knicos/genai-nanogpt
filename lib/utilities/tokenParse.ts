export default function parseTokens(text: string): string[] {
    const normalizedText = Array.from(text);
    const tokens: string[] = [];
    const regex = /(\p{P}|\p{S}|\s)/gu;

    let currentToken = '';
    for (let i = 0; i < normalizedText.length; i++) {
        const char = normalizedText[i];

        if (char === ' ') {
            tokens.push(currentToken);
            currentToken = char;
        } else if (char.match(regex)) {
            tokens.push(currentToken);
            tokens.push(char);
            currentToken = '';
        } else {
            currentToken += char;
        }
    }

    if (currentToken.length > 0) {
        // Add the last token if it exists
        tokens.push(currentToken);
    }

    return tokens.filter((t) => t.length > 0);
}
