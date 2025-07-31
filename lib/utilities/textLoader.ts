import papa from 'papaparse';

export default async function loadTextData(file: File | string) {
    return new Promise<string[]>((resolve, reject) => {
        papa.parse<string>(file, {
            header: false,
            skipEmptyLines: true,
            complete: (results) => {
                if (results.errors.length > 0) {
                    reject(new Error('Error parsing file'));
                } else {
                    resolve(results.data.slice(1).map((row) => row[0])); // Assuming each row is a single string
                }
            },
            error: (error) => {
                reject(error);
            },
        });
    });
}
