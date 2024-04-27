import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.util.Vector;
import scala.Tuple2;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class KMeansSpark{
    private static JavaSparkContext context;
    static double epsilon = 0.000001;

    public static void main(String[] args) {
        if (args.length > 5 || args.length < 4) {
            System.out.print("args should be 4 or 5: <input path> <output path> <number of clusters> <dimensions> <max number of iterations>.");
            System.exit(-1);
        }
        String input = args[0];
        String output = args[1];
        int clusters = Integer.parseInt(args[2]);
        int dimensions = Integer.parseInt(args[3]);
        int maxIterations;
        if (args.length == 5) {
            maxIterations = Integer.parseInt(args[4]);
        } else {
            maxIterations = 100;
        }
        SparkConf conf = new SparkConf().setMaster("local[*]").setAppName("KMeans");
        context = new JavaSparkContext(conf);

        JavaRDD<String> file = context.textFile(input);
        JavaRDD<Vector> data = file.map(line -> {
            String[] sValues = line.split(",");
            double[] values = new double[sValues.length - 1];
            for (int i = 0; i < dimensions ; i++) {	
                values[i] = Double.parseDouble(sValues[i]);
            }
            return new Vector(values);
        }).cache();

        long start =  System.currentTimeMillis();
        context.parallelize(KMeans(data, clusters, maxIterations)).saveAsTextFile(output);
        long end =  System.currentTimeMillis() ;

        System.out.println("***********SUCCESS************\n");
        System.out.println("Total time = "+(end - start)+" ms ");
        System.out.println("*****************************\n");

        System.exit(0);
    }
    public static List<Vector> KMeans(JavaRDD<Vector> data, int clusters, long maxIterations) {
        final List<Vector> centroids = data.takeSample(false, clusters);
        long iteration = 0;
        double tempDist = Integer.MAX_VALUE;
        Instant start = Instant.now();
        while(tempDist > epsilon && iteration < maxIterations){
            JavaPairRDD<Integer, Vector> closestData = data.mapToPair((PairFunction<Vector, Integer, Vector>) vector -> {
                int bestIndex = 0;
                double tempClosest = Double.POSITIVE_INFINITY;
                for (int i = 0; i < centroids.size(); i++) {
                    double temp = vector.squaredDist(centroids.get(i));
                    if (temp < tempClosest) {
                        tempClosest = temp;
                        bestIndex = i;
                    }
                }
                return new Tuple2<>(bestIndex, vector);
            });
            JavaPairRDD<Integer, Iterable<Vector>> pointsGroup = closestData.groupByKey();
            Map<Integer, Vector> newCentroids = pointsGroup.mapValues(
                    (Function<Iterable<Vector>, Vector>) points -> {
                        ArrayList<Vector> list = new ArrayList<>();
                        if(points != null) {
                            for(Vector point: points) {
                                list.add(point);
                            }
                        }
                        return average(list);
                    }).collectAsMap();
            tempDist = 0.0;
            for (int j = 0; j < clusters; j++) {
                tempDist += centroids.get(j).squaredDist(newCentroids.get(j));
            }
            for (Map.Entry<Integer, Vector> t : newCentroids.entrySet()) {
                centroids.set(t.getKey(), t.getValue());
            }
            iteration++;
        }
        Instant end = Instant.now();
        Duration timeElapsed = Duration.between(start, end);
        System.out.println("Time taken: " + timeElapsed.toMillis() +" milliseconds");
        System.out.println("Converged in " + iteration + " iterations.");
        System.out.println("Final centers:");
        for (Vector c : centroids) {
            System.out.println(c);
        }
        return centroids;
    }

    static Vector average(List<Vector> points) {
        int numVectors = points.size();
        Vector out = new Vector(points.get(0).elements());
        for (int i = 1; i < numVectors; i++) {
            out.addInPlace(points.get(i));
        }
        return out.divide(numVectors);
    }
}
