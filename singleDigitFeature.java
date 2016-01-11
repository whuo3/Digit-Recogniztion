import java.io.*;
import java.util.*;
import java.lang.Math;

public class singleDigitFeature {
    //For reading input
    static BufferedReader trainImage;
    static BufferedReader trainLabel;

    static BufferedReader testImage;
    static BufferedReader testLabel;
    
    static StringTokenizer tk;

    //A look up table for P(Fi | Y)  == Given label Y the probablity Fi is not background
    static int[][][] proTab;
    static int totalLabel;
    static int numFeatures;

    //Laplace smoothing: smooth the likelihoods to ensure that there are no zero counts
    private static int k;

    //Evaluation
    static int right = 0;
    static int wrong = 0;
    static int[][] evalArray = new int[10][2];//For testing the percentage of the accuracy of each digit.
    static int[][] evalArrayII = new int[10][10];//For building the confusion matrix
    
    //how the test example with the highest posterior probability (i.e., the most "prototypical" instance of that digit)
    static double[] typicalMaxRecord = new double[10];
    static char[][][] mostPrototypical = new char[10][][];

    private static String token (BufferedReader label) throws IOException {
        try {
            while (!tk.hasMoreTokens())
                tk = new StringTokenizer(label.readLine());
            return tk.nextToken();
        }
        catch (Exception e) {
        }
        return null;
    }

    private static void quit() throws IOException {
        System.out.flush();
        System.exit(0);
    }

    private static void init (String trainImageFile, String trainLabelFile, String testImageFile, String testLabelFile) throws FileNotFoundException{
        trainImage = new BufferedReader(new InputStreamReader(new FileInputStream(trainImageFile)));
        trainLabel = new BufferedReader(new InputStreamReader(new FileInputStream(trainLabelFile)));

        testImage = new BufferedReader(new InputStreamReader(new FileInputStream(testImageFile)));
        testLabel = new BufferedReader(new InputStreamReader(new FileInputStream(testLabelFile)));

        tk = new StringTokenizer("");

        //There are totally 28*28 = 784 features, in each features record the appear times of Fi that is foreground(not background)
        //The array entry is 785, the extra slot 0 is used to store the appear times of label Y 
    
        proTab = new int[10][785][2];
        for(int i = 0; i < 10; i++){
            for(int j = 0; j < 785; j++)
                for(int k = 0; k < 2; k++)
                    proTab[i][j][k] = 0;
        }

        for(int i = 0; i < 10; i++){
            for(int j = 0; j < 10; j++)
                evalArrayII[i][j] = 0;
        }

        for(int i = 0; i < 10; i++)
            typicalMaxRecord[i] = Double.NEGATIVE_INFINITY;

        totalLabel = 0;
        //In part 1_1 it possible feature is 2
        numFeatures = 2;
    }

    /***************** For Training ****************/

    private static void defineSmooth(int smoothK) throws IOException {
        k = smoothK;
    }

    private static int gInt(BufferedReader label) throws IOException {
        String temp = token(label);
        if(temp == null) return -1;
        return Integer.parseInt(temp);
    }

    private static int posToIndex(int x, int y){
        return x*28 + y + 1;
    }

    //proTab[labelY][posToIndex(x, y)][0] indicate it is on, proTab[labelY][posToIndex(x, y)][1] is off;
    private static void fillTab(int x, int y, char curChar, int labelY){
        if(curChar != ' ')
            proTab[labelY][posToIndex(x, y)][0]++;
        else
            proTab[labelY][posToIndex(x, y)][1]++;
    }

    //Read one digit in trainning data at a time
    private static boolean readDigitTrainImage() throws IOException{
        int labelY = gInt(trainLabel);
        if(labelY == -1) return false;
        char[][] digit = new char[28][];
        for(int i = 0; i < 28; i++){
            String line = trainImage.readLine();
            //If no more to read
            if(line == null) return false;
            digit[i] = line.toCharArray();
            for(int j = 0; j < 28; j++){
                char curChar = digit[i][j];
                fillTab(i, j, curChar, labelY);
            }
            //System.out.println(digit[i]);
        }
        //System.out.println(labelY);

        //proTab[labelY][0][0] record how many times labelY appears in the training data.
        proTab[labelY][0][0]++;
        totalLabel++;
        return true;
    }

    private static void trainData() throws IOException{
        while(readDigitTrainImage());
    }

    private static boolean readDigitTestImage() throws IOException{
        int labelY = gInt(testLabel);
        if(labelY == -1) return false;
        char[][] digit = new char[28][];
        for(int i = 0; i < 28; i++){
            String line = testImage.readLine();
            //If no more to read
            if(line == null) return false;
            digit[i] = line.toCharArray();
            //System.out.println(digit[i]);
        }

        int predictLabel = -1;
        double maxP = Double.NEGATIVE_INFINITY;
        for(int i = 0; i < 10; i++){
            double temp = Math.log(((double)proTab[i][0][0])/totalLabel);//MAP(maximum a posteriori)
            //double temp = 0;//maximum likelihood (ML)
            for(int j = 0; j < 28; j++){
                for(int k = 0; k < 28; k++){
                    if(digit[j][k] == ' ')
                        temp += Math.log(getOffP(j,k,i));
                    else
                        temp += Math.log(getOnP(j,k,i));
                }
            }
            if(temp > typicalMaxRecord[i]){
                typicalMaxRecord[i] = temp;
                mostPrototypical[i] = digit;
            } 
            predictLabel = temp > maxP? i : predictLabel;
            maxP = temp > maxP? temp : maxP;
        }

        /*For evaluation*/
        evalArray[labelY][0]++;
        evalArrayII[labelY][predictLabel]+=1;
        /*if(labelY != predictLabel){
            for(int i = 0; i < 28;i++) 
                System.out.println(digit[i]);
            System.out.println("Predicted value: " + predictLabel);
        }*/
        if(labelY == predictLabel){ right++; evalArray[labelY][1]++;}
        else wrong++;
        return true;
    }

    private static double getOnP(int x, int y, int labelY){
        int pos = posToIndex(x, y);
        return (((double)proTab[labelY][pos][0]) + k)/(proTab[labelY][pos][0] + proTab[labelY][pos][1] + k*numFeatures);
    }

    private static double getOffP(int x, int y, int labelY){
        int pos = posToIndex(x, y);
        return (((double)proTab[labelY][pos][1]) + k)/(proTab[labelY][pos][0] + proTab[labelY][pos][1] + k*numFeatures);
    }

    private static void testData() throws IOException{
        while(readDigitTestImage());
    }

    /********** evaluation ********/

    private static void evaluate(){
        for(int i = 0; i < evalArray.length; i++){
            System.out.println(i+" " + (((double)evalArray[i][1])/evalArray[i][0]));
        }
        System.out.println("Overall accuracy: "+((double)right)*100/(right + wrong) + "%");

        System.out.println();
        System.out.println();

        System.out.println("Confusion Table:");
        System.out.print("\t");
        for(int i = 0; i < 10; i++)
            System.out.print(i+"\t");
        System.out.println();
        for(int i = 0; i < 10; i++){
            System.out.print(i+"\t");
            for(int j = 0; j < 10; j++)
                System.out.printf("%.3f\t",(((double)evalArrayII[i][j])/evalArray[i][0]));
            System.out.println();
        }

        /*System.out.println();
        System.out.println();

        for(int i = 0; i < 10; i++){
            System.out.println("Most prototypical image for " + i + " :");
            for(int j = 0; j < 28; j++){
                for(int k = 0; k < 28; k++){
                    System.out.print(mostPrototypical[i][j][k]);
                }
                System.out.println();
            }
            System.out.println();
            System.out.println();
        }*/
    }

    private static double oddsRatio(int labelOne, int labelTwo){
        double difference = 0.0;
        for(int i = 0; i < 28; i++){
            for(int j = 0; j < 28; j++){
                difference += Math.abs(Math.log(getOnP(i, j, labelOne)) - Math.log(getOnP(i,j,labelTwo)));
                //System.out.printf("%5.1f",(Math.log(getOnP(i, j, labelOne)) - Math.log(getOnP(i,j,labelTwo)))); 
            }
            //System.out.println();
        }
        return difference;
    }

    private static void getHighestConfusion(){
        ArrayList<Double> col = new ArrayList<Double>();
        ArrayList<int[]> result = new ArrayList<int[]>();
        HashMap<Double, int[]> helper = new HashMap<Double, int[]>();
        for(int i = 0; i < 10; i++){
            for(int j = i + 1; j <10; j++){
                double diff = oddsRatio(i, j);
                //System.out.println(i + " " + j + " : " + diff);
                col.add(diff);
                helper.put(diff, new int[]{i, j});
            }
        }
        Collections.sort(col);
        System.out.println("Highest confusion are(In descending order):");
        for(int i = 0; i < 4; i++){
            int l1 = helper.get(col.get(i))[0];
            int l2 = helper.get(col.get(i))[1];
            System.out.println(l1 + " " + l2);
            oddsRatio(l1, l2);
            System.out.println();
            System.out.println();
        }
        System.out.println();
        System.out.println();   
    }

    /*********** Main *************/

    public static void main (String [] args) throws Exception {
        init(args[0], args[1], args[3], args[4]);
        defineSmooth(Integer.parseInt(args[2])); 
        trainData();
        testData();
        getHighestConfusion();
        evaluate();
        quit();
    }
}