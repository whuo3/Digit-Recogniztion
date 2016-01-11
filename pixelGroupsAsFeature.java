import java.io.*;
import java.util.*;
import java.lang.Math;

public class pixelGroupsAsFeature {
    //For reading input
    static BufferedReader trainImage;
    static BufferedReader trainLabel;

    static BufferedReader testImage;
    static BufferedReader testLabel;
    
    static StringTokenizer tk;

    //pixel Group Size n*m;
    static int n;
    static int m;
    static int numVal;
    static int numFeaRow;
    static int numFeaCol;
    static boolean disjoint;
    static boolean overlap;

    //A look up table for P(Fi | Y)  == Given label Y the probablity Fi is not background
    static int[][][] proTab;
    static int[][] proFeaTab;
    static int[] classCounter;
    static int totalTraining;
    static int numFeatures;

    //Laplace smoothing: smooth the likelihoods to ensure that there are no zero counts
    private static double smoothK;

    //Evaluation
    static int right = 0;
    static int wrong = 0;
    static int[][] evalArray = new int[10][2];

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

    private static void initProTab(int numFeatures){
        proTab = new int[10][numFeatures][numVal];
        for(int i = 0; i < 10; i++){
            for(int j = 0; j < numFeatures; j++){
                for(int k = 0; k < numVal; k++){
                    proTab[i][j][k] = 0;
                }
            }
        }
    }

    private static void initProFeaTab(int numFeatures){
        proFeaTab = new int[10][numFeatures];
        for(int i = 0; i < 10; i++){
            for(int j = 0; j < numFeatures; j++){
                proFeaTab[i][j] = 0;
            }
        }
    }

    private static void init (String trainImageFile, String trainLabelFile, String testImageFile, String testLabelFile) throws FileNotFoundException{
        trainImage = new BufferedReader(new InputStreamReader(new FileInputStream(trainImageFile)));
        trainLabel = new BufferedReader(new InputStreamReader(new FileInputStream(trainLabelFile)));

        testImage = new BufferedReader(new InputStreamReader(new FileInputStream(testImageFile)));
        testLabel = new BufferedReader(new InputStreamReader(new FileInputStream(testLabelFile)));

        tk = new StringTokenizer("");

        for(int i = 0; i < 10; i++){
            evalArray[i][0] = 0;
            evalArray[i][1] = 0;
        }

        classCounter = new int[10];
        for(int i = 0; i < 10; i++)
            classCounter[i] = 0;

        //If it is overlap, there are 27*27 features, each feature contains 2^(n*m) value
        //else if it is disjoint, there are 14*14 features, each feature contains 2^(n*m) value

        totalTraining = 0;
        numVal = (int)Math.pow(2.0, n*m);


        numFeaRow = disjoint?28/n : (28-n+1);
        numFeaCol = disjoint?28/m : (28-m+1);

        numFeatures = numFeaRow*numFeaCol;
        initProTab(numFeatures);
        initProFeaTab(numFeatures);
        
    }

    private static void setGroupSize(int _n, int _m){
        n = _n;
        m = _m;
    }

    private static void setAsDisjoint(boolean _disjoint){
        disjoint = _disjoint;
        overlap = !_disjoint;
    }

    private static void setAsOverlap(boolean _overlap){
        disjoint = !_overlap;
        overlap = _overlap;
    }    

    private static void defineSmooth(double _smoothK) throws IOException {
        smoothK = _smoothK;
    }

    private static int gInt(BufferedReader label) throws IOException {
        String temp = token(label);
        if(temp == null) return -1;
        return Integer.parseInt(temp);
    }

    private static int getValIndex(char[][] digit, int x, int y){
        int index = 0;
        for(int i = x; i < x+n; i++){
            for(int j = y; j < y+m; j++){
                if(digit[i][j] == ' ')
                    index *=2;
                else
                    index = index*2+1;
            }
        }
        return index;
    }

    private static int getFeatureIndex(int x, int y){
        if(disjoint) return x/n*numFeaCol+y/m;
        else return x*numFeaCol+y;
    }

    /***************** For Training ****************/

    private static void fillTab(char[][] digit, int x, int y, int labelY){
        //Here assume the digit.lengt%n == 0 $$ digit[].length%m ==0
        int feaIndex = getFeatureIndex(x, y);
        int valIndex = getValIndex(digit, x, y);
        //System.out.print(feaIndex + " ");
        proTab[labelY][feaIndex][valIndex]++;
        proFeaTab[labelY][feaIndex]++;
    }

    //Read one digit in trainning data at a time
    private static boolean readDigitTrainImage() throws IOException{
        int labelY = gInt(trainLabel);
        if(labelY == -1) return false;
        char[][] digit = new char[28][];

        for(int i = 0; i < 28; i++){
            String line = trainImage.readLine();
            if(line == null) return false;//If no more to read
            digit[i] = line.toCharArray();
        }

        if(disjoint){
            for(int i = 0; i < 28; i+=n){
                //System.out.println(i);
                for(int j = 0; j < 28; j+=m){
                    fillTab(digit, i, j, labelY);
                }
                //System.out.println();
            }
        }
        else{
            for(int i = 0; i < numFeaRow; i++){
                //System.out.println(i);
                for(int j = 0; j < numFeaCol; j++){
                    fillTab(digit, i, j, labelY);
                }
                //System.out.println();
            }
        }

        classCounter[labelY]++;
        totalTraining++;

        return true;
    }

    private static void trainData() throws IOException{
        //readDigitTrainImage();
        int trainNums = 0;
        while(readDigitTrainImage()){
            trainNums++;
        };
        System.out.println("Training example sizes is " + trainNums);
    }

    /***********TestData*************/

    private static double getP(int x, int y, int labelY, char[][] digit){
        //System.out.println(smoothK);
        return ((double)proTab[labelY][getFeatureIndex(x, y)][getValIndex(digit, x, y)] + smoothK)/(proFeaTab[labelY][getFeatureIndex(x, y)] + numFeatures*smoothK);
    }
    
    private static boolean readDigitTestImage() throws IOException{
        int labelY = gInt(testLabel);
        if(labelY == -1) return false;
        char[][] digit = new char[28][];
        for(int i = 0; i < 28; i++){
            String line = testImage.readLine();
            if(line == null) return false;
            //System.out.println(line);
            digit[i] = line.toCharArray();
        }

        int predictLabel = -1;
        double maxP = Double.NEGATIVE_INFINITY;
        
        if(disjoint){
            for(int k = 0; k < 10; k++){
                double temp = Math.log(((double)classCounter[k])/totalTraining);
                for(int i = 0; i < 28; i+=n){
                    for(int j = 0; j < 28; j+=m){
                        temp += Math.log(getP(i,j,k, digit));
                    }
                }
                maxP = maxP > temp? maxP:temp;
                predictLabel = maxP > temp? predictLabel:k;
            }
        }
        else{
            for(int k = 0; k < 10; k++){
                double temp = Math.log(((double)classCounter[k])/totalTraining);
                for(int i = 0; i < numFeaRow; i++){
                    for(int j = 0; j < numFeaCol; j++){
                        temp += Math.log(getP(i,j,k,digit));
                    }
                }
                maxP = maxP > temp? maxP:temp;
                predictLabel = maxP > temp? predictLabel:k;
            }  
        }

        //System.out.println("Predicted value: " + predictLabel);
        evalArray[labelY][0]++;
        if(labelY == predictLabel){ right++; evalArray[labelY][1]++;}
        else wrong++;
        return true;
    }

    private static void testData() throws IOException{
        while(readDigitTestImage());
    }
    

    /********** evaluation ********/

    private static void evaluate(){
        for(int i = 0; i < evalArray.length; i++){
            System.out.println(i+" " + (((double)evalArray[i][1])*100/evalArray[i][0]));
        }
        System.out.println(((double)right)*100/(right + wrong) + "%");
    }
    


    /*********** Main *************/

    public static void main (String [] args) throws Exception {
        setGroupSize(2,3);
        //setAsDisjoint(true);
        setAsOverlap(true);

        init(args[0], args[1], args[3], args[4]);
        defineSmooth(Integer.parseInt(args[2])); 
        trainData();
        testData();
        if(disjoint) System.out.println("disjoint");
        else System.out.println("Overlap");
        System.out.println("n = " + n + "; m = " + m + ";");
        evaluate();

        quit();
    }
}