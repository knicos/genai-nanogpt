export default function parseTokens(text: string): string[] {
    const normalizedText = Array.from(text);
    const tokens: string[] = [];
    const regex = /(\p{P}|\p{S}|\s)/gu;

    let currentToken = '';
    for (let i = 0; i < normalizedText.length; i++) {
        const char = normalizedText[i];

        if (char === ' ') {
            const charNext = normalizedText[i + 1] ?? '';
            if (charNext !== ' ') {
                tokens.push(currentToken);
                currentToken = char;
            } else {
                currentToken += char;
            }
        } else if (char.match(regex)) {
            tokens.push(currentToken);

            // Repeated identical punctuation should be merged
            let charString = char;
            while (i + 1 < normalizedText.length && normalizedText[i + 1] === char) {
                charString += normalizedText[i + 1];
                i++;
            }
            tokens.push(charString);
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
