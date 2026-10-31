using System;
using System.IO;
using System.Text;
using System.Globalization;
using System.Collections.Generic;

public class NwResult {
    public int Length, Positives;
}

// Strict global Needleman-Wunsch, linear gap cost, deterministic traceback.
public static class NwEngine {
    static readonly string Alphabet = "ARNDCQEGHILKMFPSTWYV";
    static int[,] Matrix;
    static readonly int[] Index = new int[128];
    public static void LoadMatrix(string path) {
        for (int i=0;i<128;i++) Index[i]=-1;
        for (int i=0;i<Alphabet.Length;i++) Index[Alphabet[i]]=i;
        Matrix=new int[20,20];
        var lines=File.ReadAllLines(path);
        for(int i=0;i<20;i++) {
            var xs=lines[i].Split(new char[]{' ','\t'}, StringSplitOptions.RemoveEmptyEntries);
            if(xs.Length!=21 || xs[0]!=Alphabet[i].ToString()) throw new Exception("Matrix format");
            for(int j=0;j<20;j++) Matrix[i,j]=int.Parse(xs[j+1]);
        }
        for(int i=0;i<20;i++) for(int j=0;j<20;j++)
            if(Matrix[i,j]!=Matrix[j,i]) throw new Exception("Asymmetric matrix");
    }
    public static NwResult Align(string train, string test) {
        const int gap=-5;
        int m=train.Length,n=test.Length,w=n+1;
        var a=new int[m]; var b=new int[n];
        for(int i=0;i<m;i++) {a[i]=Index[train[i]];if(a[i]<0)throw new Exception("Invalid train residue");}
        for(int j=0;j<n;j++) {b[j]=Index[test[j]];if(b[j]<0)throw new Exception("Invalid test residue");}
        var prev=new int[w];var curr=new int[w];var trace=new byte[(m+1)*w];
        for(int j=1;j<=n;j++){prev[j]=j*gap;trace[j]=2;}
        for(int i=1;i<=m;i++) {
            curr[0]=i*gap;trace[i*w]=1;
            for(int j=1;j<=n;j++) {
                int best=prev[j-1]+Matrix[a[i-1],b[j-1]];byte dir=0;
                int up=prev[j]+gap, left=curr[j-1]+gap;
                if(up>best){best=up;dir=1;}
                if(left>best){best=left;dir=2;}
                curr[j]=best;trace[i*w+j]=dir;
            }
            var temp=prev;prev=curr;curr=temp;
        }
        var result=new NwResult();
        int p=m,q=n,k=0;
        while(p>0 || q>0) {
            byte dir=trace[p*w+q];
            if(dir==0){
                char x=train[--p],y=test[--q];
                if(Matrix[Index[x],Index[y]]>0)result.Positives++;
            } else if(dir==1){p--;}
            else {q--;}
            k++;
        }
        result.Length=k;
        return result;
    }
    public static void Run(string input, string output) {
        var groups = new Dictionary<string, List<string[]>>();
        var ids = new HashSet<string>();
        foreach (var line in File.ReadAllLines(input)) {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var xs = line.Split('\t');
            if (xs.Length != 3 || xs[2].Length == 0 || !ids.Add(xs[1]))
                throw new Exception("Invalid or duplicate input sequence");
            foreach (char aa in xs[2])
                if (Alphabet.IndexOf(aa) < 0) throw new Exception("Invalid residue");
            if (!groups.ContainsKey(xs[0])) groups[xs[0]] = new List<string[]>();
            groups[xs[0]].Add(xs);
        }
        var pairs = new string[][] { new string[] {"S8754", "S669"},
                                    new string[] {"S4346", "S571"} };
        foreach (var pair in pairs) {
            string filename = pair[0] + "_vs_" + pair[1] + "_similarity_pct.csv";
            using (var f = new StreamWriter(Path.Combine(output, filename), false,
                                           new UTF8Encoding(true))) {
                var header = new List<string> { "test_uid" };
                foreach (var train in groups[pair[0]]) header.Add(train[1]);
                f.WriteLine(string.Join(",", header.ToArray()));
                int count = 0;
                foreach (var test in groups[pair[1]]) {
                    var row = new List<string> { test[1] };
                    foreach (var train in groups[pair[0]]) {
                        var result = Align(train[2], test[2]);
                        row.Add((100.0 * result.Positives / result.Length)
                                .ToString("F10", CultureInfo.InvariantCulture));
                    }
                    f.WriteLine(string.Join(",", row.ToArray()));
                    count++;
                    if (count % 10 == 0)
                        Console.WriteLine(pair[0] + " vs " + pair[1] + ": " + count
                                          + " / " + groups[pair[1]].Count);
                }
            }
        }
    }
}
