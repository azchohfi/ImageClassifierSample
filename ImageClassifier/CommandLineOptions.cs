using CommandLine;

namespace ImageClassifier
{
    public class CommandLineOptions
    {
        [Value(index: 0, Required = true, HelpText = "Image file Path to analyze.")]
        public string Path { get; set; }

        [Option(shortName: 'c', longName: "confidence", Required = false, HelpText = "Minimum confidence.", Default = 0.9f)]
        public float Confidence { get; set; }
    }
}
