import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;

public class KMeans {

    public static void writeCentroids(double[][] points, int index, String dir, String[] classes) {
        try {
            FileSystem hdfs = FileSystem.get(new Configuration());
            Path centroidFile = new Path(dir + "/centroid" + index);
            OutputStreamWriter outStreamWriter = new OutputStreamWriter(hdfs.create(centroidFile));
            for (int i = 0; i < points.length; i++) {
                for (int j = 0; j < points[i].length; j++) {
                    outStreamWriter.append(Double.toString(points[i][j])).append(",");
                }
                classes[i] = !classes[i].isEmpty() ? classes[i].substring(0, classes[i].length() - 1) : classes[i];
                outStreamWriter.append(classes[i]).append("\n");
            }
            outStreamWriter.flush();
            outStreamWriter.close();
        } catch (Exception e) {
            System.out.println("error updating the centroid");
        }

    }

    public static double[][] readCentroids(int clusters, int index, String dir, int dimensions) {
        double[][] centroid = new double[clusters][dimensions];
        try {
            FileSystem hdfs = FileSystem.get(new Configuration());
            Path centroidFile = new Path(dir + "/centroid" + (index - 1));
            InputStreamReader inStreamReader = new InputStreamReader(hdfs.open(centroidFile));
            BufferedReader bufferReader = new BufferedReader(inStreamReader);
            String line = bufferReader.readLine();
            int lineNumber = 0;
            while (line != null) {
                String[] values = line.split(",");
                for (int i = 0; i < values.length - 1; i++) {
                    centroid[lineNumber][i] = Double.parseDouble(values[i]);
                }
                line = bufferReader.readLine();
                lineNumber++;
            }
            bufferReader.close();
        } catch (Exception e) {
            System.out.println("error opening the file");
            return null;
        }
        return centroid;
    }

    public static boolean converge(int clusters, int index, String dir, int dimensions, int maxIterations) {
        if (index == 0) {
            return false;
        } else if (index == maxIterations) {
            return true;
        }
        double[][] centroid1 = readCentroids(clusters, index++, dir, dimensions);
        double[][] centroid2 = readCentroids(clusters, index, dir, dimensions);
        for (int i = 0; i < clusters; i++) {
            double distance = 0;
            for (int j = 0; j < centroid1[i].length; j++) {
                distance += Math.pow(centroid1[i][j] - centroid2[i][j], 2);
            }
            distance = Math.sqrt(distance);
            double threshold = Math.pow(0.01, 2) * clusters * dimensions;
            threshold = Math.sqrt(threshold);
            if (distance > threshold) {
                return false;
            }
        }
        return true;
    }

    public static void main(String[] args) throws Exception {
        if (args.length > 5 || args.length < 4) {
            System.out.print("args should be 4 or 5: <input path> <output path> <number of clusters> <dimensions> <max number of iterations>.");
            System.exit(-1);
        }
        String input = args[0];
        String output = args[1];
        String clusters = args[2];
        String dimensions = args[3];
        String maxIterations;
        if (args.length == 5) {
            maxIterations = args[4];
        } else {
            maxIterations = "100";
        }

        int iteration = 0;
        long start = System.currentTimeMillis();

        Configuration initConf = new Configuration();
        initConf.set("Clusters", clusters);
        initConf.set("Dimensions", dimensions);
        initConf.set("Output", output);

        Job initJob = Job.getInstance(initConf, "InitKMeans");
        initJob.setJarByClass(KMeans.class);
        initJob.setMapperClass(KMeansInit.Map.class);
        initJob.setReducerClass(KMeansInit.Reduce.class);

        initJob.setMapOutputKeyClass(IntWritable.class);
        initJob.setMapOutputValueClass(TextArrayWritable.class);

        initJob.setOutputKeyClass(Text.class);
        initJob.setOutputValueClass(Text.class);

        initJob.setInputFormatClass(TextInputFormat.class);
        initJob.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(initJob, new Path(input));
        FileOutputFormat.setOutputPath(initJob, new Path(output + "/iteration" + iteration));

        initJob.waitForCompletion(true);

        Configuration conf = new Configuration();
        while (!converge(Integer.parseInt(clusters), iteration, output, Integer.parseInt(dimensions), Integer.parseInt(maxIterations))) {
            iteration++;
            conf.set("Iteration", Integer.toString(iteration));
            conf.set("Clusters", clusters);
            conf.set("Dimensions", dimensions);
            conf.set("Output", output);
            Job job = Job.getInstance(conf, "KMeans");
            job.setJarByClass(KMeans.class);
            job.setMapperClass(KMeansIteration.Map.class);
            job.setReducerClass(KMeansIteration.Reduce.class);

            job.setMapOutputKeyClass(IntWritable.class);
            job.setMapOutputValueClass(TextArrayWritable.class);

            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);

            job.setInputFormatClass(TextInputFormat.class);
            job.setOutputFormatClass(TextOutputFormat.class);

            Path jobInput = new Path(output + "/iteration" + (iteration - 1) + "/part-r-00000");
            Path jobOutput = new Path(output + "/iteration" + iteration);
            FileInputFormat.addInputPath(job, jobInput);
            FileOutputFormat.setOutputPath(job, jobOutput);
            job.waitForCompletion(true);
        }

        long end = System.currentTimeMillis();
        System.out.println("Time taken: " + (end - start) + " ms");
    }

    public static class TextArrayWritable extends ArrayWritable {
        public TextArrayWritable() {
            super(Text.class);
        }

        public TextArrayWritable(String[] strings) {
            super(Text.class);
            Text[] texts = new Text[strings.length];
            for (int i = 0; i < strings.length; i++) {
                texts[i] = new Text(strings[i]);
            }
            set(texts);
        }

        public TextArrayWritable(double[] doubles) {
            super(Text.class);
            Text[] texts = new Text[doubles.length];
            for (int i = 0; i < doubles.length; i++) {
                texts[i] = new Text(String.valueOf(doubles[i]));
            }
            set(texts);
        }
    }


    public static class KMeansInit {
        public static class Map extends Mapper<LongWritable, Text, IntWritable, TextArrayWritable> {
            private final Random random = new Random();
            private int clusters;
            private int dimensions;

            @Override
            protected void setup(Context context) {
                Configuration config = context.getConfiguration();
                clusters = Integer.parseInt(config.get("Clusters"));
                dimensions = Integer.parseInt(config.get("Dimensions"));
            }

            @Override
            public void map(LongWritable id, Text value, Context context) throws IOException, InterruptedException {
                String line = value.toString();
                String[] values = line.split(",");
                if (values.length == dimensions + 1) {
                    context.write(new IntWritable(1 + random.nextInt(clusters)), new TextArrayWritable(values));
                }
            }
        }

        public static class Reduce extends Reducer<IntWritable, TextArrayWritable, Text, Text> {

            private final int iteration = 0;
            private int clusters;
            private int dimensions;
            private String output = "";

            private double[][] centroid;

            private String[] classes;
            private String lastPoint = "";

            @Override
            protected void setup(Context context) {
                Configuration config = context.getConfiguration();
                clusters = Integer.parseInt(config.get("Clusters"));
                dimensions = Integer.parseInt(config.get("Dimensions"));
                output = config.get("Output");
                centroid = new double[clusters][dimensions];
                classes = new String[clusters];
                Arrays.fill(classes, "");
            }

            @Override
            public void reduce(IntWritable id, Iterable<TextArrayWritable> values, Context context) throws IOException, InterruptedException {
                int totalPoints = 0;
                double[] center = new double[dimensions];
                Arrays.fill(center, 0);
                int key = id.get() - 1;
                for (TextArrayWritable value : values) {
                    String[] point = value.toStrings();
                    if (!classes[key].contains(point[dimensions])) {
                        classes[key] = classes[key] + point[dimensions] + "+";
                    }
                    totalPoints++;

                    String data = "";
                    for (int i = 0; i < point.length - 1; i++) {
                        data += point[i] + ",";
                        center[i] += Double.parseDouble(point[i]);
                    }
                    lastPoint = data;
                    context.write(new Text(id + ","), new Text(data + point[dimensions]));
                }
                for (int i = 0; i < center.length; i++) {
                    centroid[key][i] = center[i] / totalPoints;
                }

            }

            @Override
            protected void cleanup(Context context) {
                Configuration config = context.getConfiguration();
                config.set("lastPoint", lastPoint);
                writeCentroids(centroid, iteration, output, classes);
            }
        }


    }

    public static class KMeansIteration {
        public static class Map extends Mapper<LongWritable, Text, IntWritable, TextArrayWritable> {

            private int clusters;
            private int dimensions;
            private int iteration = 0;
            private double[][] centroid;
            private String output = "";

            @Override
            protected void setup(Context context) {
                Configuration config = context.getConfiguration();
                clusters = Integer.parseInt(config.get("Clusters"));
                dimensions = Integer.parseInt(config.get("Dimensions"));
                iteration = Integer.parseInt(config.get("Iteration"));
                output = config.get("Output");
                centroid = readCentroids(clusters, iteration, output, dimensions);
            }

            @Override
            public void map(LongWritable id, Text value, Context context) throws IOException, InterruptedException {
                double[] point = new double[dimensions];
                String[] values = value.toString().split(",");
                int i = 0;
                values[1] = values[1].replaceAll("\\s+", " ");
                for (i = 0; i < dimensions; i++) {
                    point[i] = Double.parseDouble(values[i + 1]);
                }
                int clusterId = 0;
                double min = Double.MAX_VALUE;
                double distance;
                for (i = 0; i < centroid.length; i++) {
                    distance = 0;
                    for (int j = 0; j < dimensions; j++) {
                        distance += Math.pow((centroid[i][j] - point[j]), 2);
                    }
                    distance = Math.sqrt(distance);
                    if (distance < min) {
                        min = distance;
                        clusterId = i + 1;
                    }
                }
                context.write(new IntWritable(clusterId), new TextArrayWritable(Arrays.copyOfRange(values, 1, values.length)));
            }
        }

        public static class Reduce extends Reducer<IntWritable, TextArrayWritable, Text, Text> {
            private int clusters;
            private int dimensions;
            private int iteration = 0;
            private String output = "";

            private double[][] centroid;
            private String[] classes;

            private String lastPoint;

            @Override
            protected void setup(Context context) {
                Configuration config = context.getConfiguration();
                clusters = Integer.parseInt(config.get("Clusters"));
                dimensions = Integer.parseInt(config.get("Dimensions"));
                iteration = Integer.parseInt(config.get("Iteration"));
                output = config.get("Output");
                centroid = new double[clusters][dimensions];
                output = config.get("Output");
                classes = new String[clusters];
                Arrays.fill(classes, "");
                lastPoint = config.get("lastPoint");
            }

            @Override
            public void reduce(IntWritable id, Iterable<TextArrayWritable> values, Context context) throws IOException, InterruptedException {
                double[] center = new double[dimensions];
                int key = id.get() - 1;
                Iterator<TextArrayWritable> iterator = values.iterator();
                if (!iterator.hasNext()) {
                    String[] d = lastPoint.split(",");
                    for (int i = 0; i < d.length; i++) {
                        centroid[key][i] = Double.parseDouble(d[i]);
                    }
                    context.write(new Text(id + ","), new Text(lastPoint));
                } else {
                    Arrays.fill(center, 0);
                    int totalPoints = 0;
                    for (TextArrayWritable value : values) {
                        String[] point = value.toStrings();
                        if (!classes[key].contains(point[dimensions])) {
                            classes[key] = classes[key] + point[dimensions] + "+";
                        }
                        totalPoints++;
                        String data = "";
                        for (int i = 0; i < point.length - 1; i++) {
                            data += point[i] + ",";
                            center[i] += Double.parseDouble(point[i]);
                        }
                        lastPoint = data + point[dimensions];
                        context.write(new Text(id + ","), new Text(data + point[dimensions]));
                    }
                    for (int i = 0; i < center.length; i++) {
                        centroid[key][i] = center[i] / totalPoints;
                    }
                }
            }

            @Override
            protected void cleanup(Context context) {
                Configuration config = context.getConfiguration();
                config.set("lastPoint", lastPoint);
                writeCentroids(centroid, iteration, output, classes);
            }
        }
    }

}
