package org.lemming.tests;

import java.util.Random;
import java.util.concurrent.Callable;

import org.apache.commons.math3.exception.MathIllegalStateException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.stat.descriptive.StatisticalSummary;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.FastMath;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

@SuppressWarnings("static-method")
public class FastMathTestPerformance {
	
	private static final int RUNS = Integer.parseInt(System.getProperty("testRuns","10000000"));
    private static final double F1 = 1d / RUNS;

    // Header format
    private static final String FMT_HDR = "%-5s %13s %13s %13s Runs=%d Java %s (%s) %s (%s)";
    // Detail format
    private static final String FMT_DTL = "%-5s %6d %6.1f %6d %6.4f %6d %6.4f";
    /** Nanoseconds to milliseconds conversion factor ({@value}). */
    private static final double NANO_TO_MILLI = 1e-6;
    /** RNG. */
    private static final Random rng = new Random();


	@BeforeClass
	public static void setUpBeforeClass() {
		System.out.println(String.format(FMT_HDR,
                "Name","StrictMath","FastMath","Math",RUNS,
                System.getProperty("java.version"),
                System.getProperty("java.runtime.version","?"),
                System.getProperty("java.vm.name"),
                System.getProperty("java.vm.version")
                ));
	}
	
	private static void report(String name, long strictMathTime, long fastMathTime, long mathTime) {
		System.out.println(String.format(FMT_DTL,
                name,
                strictMathTime / RUNS, (double) strictMathTime / strictMathTime,
                fastMathTime / RUNS, (double) fastMathTime / strictMathTime,
                mathTime / RUNS, (double) mathTime / strictMathTime
                ));
    }

	
		@Test
	    public void testLog() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.log(0.01 + i);
	        }
	        long strictMath = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.log(0.01 + i);
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.log(0.01 + i);
	        }
	        long mathTime = System.nanoTime() - time;

	        report("log",strictMath,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }

	    @Test
	    public void testLog10() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.log10(0.01 + i);
	        }
	        long strictMath = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.log10(0.01 + i);
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.log10(0.01 + i);
	        }
	        long mathTime = System.nanoTime() - time;

	        report("log10",strictMath,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }

	    @Test
	    public void testLog1p() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.log1p(-0.9 + i);
	        }
	        long strictMath = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.log1p(-0.9 + i);
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.log1p(-0.9 + i);
	        }
	        long mathTime = System.nanoTime() - time;

	        report("log1p",strictMath,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }

	    @Test
	    public void testPow() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.pow(0.01 + i * F1, 2);
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.pow(0.01 + i * F1, 2);
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.pow(0.01 + i * F1, 2);
	        }
	        long mathTime = System.nanoTime() - time;
	        report("pow",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }

	    @Test
	    public void testExp() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.exp(100 * i * F1);
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.exp(100 * i * F1);
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.exp(100 * i * F1);
	        }
	        long mathTime = System.nanoTime() - time;

	        report("exp",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }

	    @Test
	    public void testSin() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.sin(100 * (i - RUNS/2) * F1);
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.sin(100 * (i - RUNS/2) * F1);
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.sin(100 * (i - RUNS/2) * F1);
	        }
	        long mathTime = System.nanoTime() - time;

	        report("sin",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }

	    @Test
	    public void testAsin() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.asin(0.999 * (i - RUNS/2) * F1);
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.asin(0.999 * (i - RUNS/2) * F1);
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.asin(0.999 * (i - RUNS/2) * F1);
	        }
	        long mathTime = System.nanoTime() - time;

	        report("asin",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }

	    @Test
	    public void testCos() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.cos(100 * (i - RUNS/2) * F1);
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.cos(100 * (i - RUNS/2) * F1);
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.cos(100 * (i - RUNS/2) * F1);
	        }
	        long mathTime = System.nanoTime() - time;

	        report("cos",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }
	            
	    @Test
	    public void testAcos() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.acos(0.999 * (i - RUNS/2) * F1);
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.acos(0.999 * (i - RUNS/2) * F1);
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.acos(0.999 * (i - RUNS/2) * F1);
	        }
	        long mathTime = System.nanoTime() - time;
	        report("acos",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }

	    @Test
	    public void testTan() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.tan(100 * (i - RUNS/2) * F1);
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.tan(100 * (i - RUNS/2) * F1);
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.tan(100 * (i - RUNS/2) * F1);
	        }
	        long mathTime = System.nanoTime() - time;

	        report("tan",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }

	    @Test
	    public void testAtan() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.atan(100 * (i - RUNS/2) * F1);
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.atan(100 * (i - RUNS/2) * F1);
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.atan(100 * (i - RUNS/2) * F1);
	        }
	        long mathTime = System.nanoTime() - time;

	        report("atan",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }

	    @Test
	    public void testAtan2() {
	        double x = 0;
	        long time = System.nanoTime();
	        int max   = (int) FastMath.floor(FastMath.sqrt(RUNS));
	        for (int i = 0; i < max; i++) {
	            for (int j = 0; j < max; j++) {
	                x += StrictMath.atan2((i - max/2) * (100.0 / max), (j - max/2) * (100.0 / max));
	            }
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < max; i++) {
	            for (int j = 0; j < max; j++) {
	                x += FastMath.atan2((i - max/2) * (100.0 / max), (j - max/2) * (100.0 / max));
	            }
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < max; i++) {
	            for (int j = 0; j < max; j++) {
	                x += Math.atan2((i - max/2) * (100.0 / max), (j - max/2) * (100.0 / max));
	            }
	        }
	        long mathTime = System.nanoTime() - time;

	        report("atan2",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }

	    @Test
	    public void testHypot() {
	        double x = 0;
	        long time = System.nanoTime();
	        int max   = (int) FastMath.floor(FastMath.sqrt(RUNS));
	        for (int i = 0; i < max; i++) {
	            for (int j = 0; j < max; j++) {
	                x += StrictMath.atan2((i - max/2) * (100.0 / max), (j - max/2) * (100.0 / max));
	            }
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < max; i++) {
	            for (int j = 0; j < max; j++) {
	                x += FastMath.atan2((i - max/2) * (100.0 / max), (j - max/2) * (100.0 / max));
	            }
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < max; i++) {
	            for (int j = 0; j < max; j++) {
	                x += Math.atan2((i - max/2) * (100.0 / max), (j - max/2) * (100.0 / max));
	            }
	        }
	        long mathTime = System.nanoTime() - time;

	        report("hypot",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }
	     
	    @Test
	    public void testCbrt() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.cbrt(100 * i * F1);
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.cbrt(100 * i * F1);
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.cbrt(100 * i * F1);
	        }
	        long mathTime = System.nanoTime() - time;

	        report("cbrt",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }

	    @Test
	    public void testSqrt() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.sqrt(100 * i * F1);
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.sqrt(100 * i * F1);
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.sqrt(100 * i * F1);
	        }
	        long mathTime = System.nanoTime() - time;

	        report("sqrt",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }

	    @Test
	    public void testCosh() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.cosh(100 * (i - RUNS/2) * F1);
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.cosh(100 * (i - RUNS/2) * F1);
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.cosh(100 * (i - RUNS/2) * F1);
	        }
	        long mathTime = System.nanoTime() - time;

	        report("cosh",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }

	    @Test
	    public void testSinh() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.sinh(100 * (i - RUNS/2) * F1);
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.sinh(100 * (i - RUNS/2) * F1);
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.sinh(100 * (i - RUNS/2) * F1);
	        }
	        long mathTime = System.nanoTime() - time;

	        report("sinh",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }

	    @Test
	    public void testTanh() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.tanh(100 * (i - RUNS/2) * F1);
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.tanh(100 * (i - RUNS/2) * F1);
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.tanh(100 * (i - RUNS/2) * F1);
	        }
	        long mathTime = System.nanoTime() - time;

	        report("tanh",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }
	     
	    @Test
	    public void testExpm1() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.expm1(100 * (i - RUNS/2) * F1);
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.expm1(100 * (i - RUNS/2) * F1);
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.expm1(100 * (i - RUNS/2) * F1);
	        }
	        long mathTime = System.nanoTime() - time;
	        report("expm1",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }

	    @Test
	    public void testAbs() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.abs(i * (1 - 0.5 * RUNS));
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.abs(i * (1 - 0.5 * RUNS));
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.abs(i * (1 - 0.5 * RUNS));
	        }
	        long mathTime = System.nanoTime() - time;

	        report("abs",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }
	    
	    @Test
	    public void testRound() {
	        long x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.round(i * (1 - 0.333 * RUNS));
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.round(i * (1 - 0.333 * RUNS));
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.round(i * (1 - 0.333 * RUNS));
	        }
	        long mathTime = System.nanoTime() - time;

	        report("round",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }
	    
	    @Test
	    public void testCeil() {
	        double x = 0;
	        long time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += StrictMath.ceil(i * (1 + 0.333 * RUNS));
	        }
	        long strictTime = System.nanoTime() - time;

	        x = 0;
	        //long y=0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += FastMath.ceil(i * (1 + 0.333 * RUNS));
	        	//y+= (long) (i * (1 + 0.333 * RUNS) + 0.5);
	        }
	        long fastTime = System.nanoTime() - time;

	        x = 0;
	        time = System.nanoTime();
	        for (int i = 0; i < RUNS; i++) {
	            x += Math.ceil(i * (1 + 0.333 * RUNS));
	        }
	        long mathTime = System.nanoTime() - time;

	        report("ceil",strictTime,fastTime,mathTime);
	        Assert.assertTrue(!Double.isNaN(x));
	    }

	    @SuppressWarnings("boxing")
	    @Test
	    public void testSimpleBenchmark() {
	        final String SM = "StrictMath";
	        final String M = "Math";
	        final String FM = "FastMath";

	        final int numStat = 100;
	        final int numCall = RUNS / numStat;

	        final double x = Math.random();
	        final double y = Math.random();

	        timeAndReport("log",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.log(x);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.log(x);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.log(x);
	                                        }
	                                    });

	        timeAndReport("log10",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.log10(x);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.log10(x);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.log10(x);
	                                        }
	                                    });

	        timeAndReport("log1p",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.log1p(x);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.log1p(x);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.log1p(x);
	                                        }
	                                    });

	        timeAndReport("pow",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.pow(x, y);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.pow(x, y);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.pow(x, y);
	                                        }
	                                    });

	        timeAndReport("exp",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.exp(x);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.exp(x);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.exp(x);
	                                        }
	                                    });

	        timeAndReport("sin",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.sin(x);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.sin(x);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.sin(x);
	                                        }
	                                    });

	        timeAndReport("asin",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.asin(x);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.asin(x);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.asin(x);
	                                        }
	                                    });

	        timeAndReport("cos",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.cos(x);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.cos(x);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.cos(x);
	                                        }
	                                    });

	        timeAndReport("acos",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.acos(x);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.acos(x);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.acos(x);
	                                        }
	                                    });

	        timeAndReport("tan",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.tan(x);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.tan(x);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.tan(x);
	                                        }
	                                    });

	        timeAndReport("atan",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.atan(x);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.atan(x);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.atan(x);
	                                        }
	                                    });

	        timeAndReport("atan2",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.atan2(x, y);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.atan2(x, y);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.atan2(x, y);
	                                        }
	                                    });

	        timeAndReport("hypot",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.hypot(x, y);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.hypot(x, y);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.hypot(x, y);
	                                        }
	                                    });


	        timeAndReport("cbrt",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.cbrt(x);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.cbrt(x);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.cbrt(x);
	                                        }
	                                    });

	        timeAndReport("sqrt",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.sqrt(x);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.sqrt(x);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.sqrt(x);
	                                        }
	                                    });

	        timeAndReport("cosh",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.cosh(x);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.cosh(x);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.cosh(x);
	                                        }
	                                    });

	        timeAndReport("sinh",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.sinh(x);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.sinh(x);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.sinh(x);
	                                        }
	                                    });

	        timeAndReport("tanh",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.tanh(x);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.tanh(x);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.tanh(x);
	                                        }
	                                    });

	        timeAndReport("expm1",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.expm1(x);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.expm1(x);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.expm1(x);
	                                        }
	                                    });

	        timeAndReport("abs",
	                                    numCall,
	                                    numStat,
	                                    false,
	                                    new RunTest(SM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return StrictMath.abs(x);
	                                        }
	                                    },
	                                    new RunTest(M) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return Math.abs(x);
	                                        }
	                                    },
	                                    new RunTest(FM) {
	                                        @Override
	                                        public Double call() throws Exception {
	                                            return FastMath.abs(x);
	                                        }
	                                    });
	        timeAndReport("round",
                    numCall,
                    numStat,
                    false,
                    new RunTest(SM) {
                        @Override
                        public Double call() throws Exception {
                            return (double) StrictMath.round(x);
                        }
                    },
                    new RunTest(M) {
                        @Override
                        public Double call() throws Exception {
                            return (double) Math.round(x);
                        }
                    },
                    new RunTest(FM) {
                        @Override
                        public Double call() throws Exception {
                            return (double) FastMath.round(x);
                        }
                    });
	        
	        timeAndReport("ceil",
                    numCall,
                    numStat,
                    false,
                    new RunTest(SM) {
                        @Override
                        public Double call() throws Exception {
                            return (double) StrictMath.ceil(x);
                        }
                    },
                    new RunTest(M) {
                        @Override
                        public Double call() throws Exception {
                            return (double) Math.ceil(x);
                        }
                    },
                    new RunTest(FM) {
                        @Override
                        public Double call() throws Exception {
                            return (double) FastMath.ceil(x);
                        }
                    });
	    }

	private static StatisticalSummary[] timeAndReport(String title, int repeatChunk, int repeatStat, boolean runGC, RunTest... methods) {
		// Header format.
		final String hFormat = "%s (calls per timed block: %d, timed blocks: %d, time unit: ms)";

		// Width of the longest name.
		int nameLength = 0;
		for (RunTest m : methods) {
			int len = m.getName().length();
			if (len > nameLength) {
				nameLength = len;
			}
		}
		final String nameLengthFormat = "%" + nameLength + "s";

		// Column format.
		final String cFormat = nameLengthFormat + " %14s %14s %10s %10s %15s";
		// Result format.
		final String format = nameLengthFormat + " %.8e %.8e %.4e %.4e % .8e";

		System.out.println(String.format(hFormat, title, repeatChunk, repeatStat));
		System.out.println(String.format(cFormat, "name", "time/call", "std error", "total time", "ratio", "difference"));
		final StatisticalSummary[] time = time(repeatChunk, repeatStat, runGC, methods);
		final double refSum = time[0].getSum() * repeatChunk;
		for (int i = 0, max = time.length; i < max; i++) {
			final StatisticalSummary s = time[i];
			final double sum = s.getSum() * repeatChunk;
			System.out.println(String.format(format, methods[i].getName(), s.getMean(), s.getStandardDeviation(), sum, sum / refSum, sum - refSum));
		}

		return time;
	}
	
	@SuppressWarnings("unchecked")
	private static StatisticalSummary[] time(int repeatChunk, int repeatStat, boolean runGC, Callable<Double>... methods) {
		final double[][][] times = timesAndResults(repeatChunk, repeatStat, runGC, methods);

		final int len = methods.length;
		final StatisticalSummary[] stats = new StatisticalSummary[len];
		for (int j = 0; j < len; j++) {
			final SummaryStatistics s = new SummaryStatistics();
			for (int k = 0; k < repeatStat; k++) {
				s.addValue(times[j][k][0]);
			}
			stats[j] = s.getSummary();
		}

		return stats;
	}
	
	@SuppressWarnings("unchecked")
	private static double[][][] timesAndResults(int repeatChunk, int repeatStat, boolean runGC, Callable<Double>... methods) {
		final int numMethods = methods.length;
		final double[][][] timesAndResults = new double[numMethods][repeatStat][2];

		try {
			for (int k = 0; k < repeatStat; k++) {
				for (int j = 0; j < numMethods; j++) {
					if (runGC) {
						// Try to perform GC outside the timed block.
						System.gc();
					}

					final Callable<Double> r = methods[j];
					final double[] result = new double[repeatChunk];

					// Timed block.
					final long start = System.nanoTime();
					for (int i = 0; i < repeatChunk; i++) {
						result[i] = r.call();
					}
					final long stop = System.nanoTime();

					// Collect run time.
					timesAndResults[j][k][0] = (stop - start) * NANO_TO_MILLI;
					// Keep track of a randomly selected result.
					timesAndResults[j][k][1] = result[rng.nextInt(repeatChunk)];
				}
			}
		} catch (Exception e) {
			// Abort benchmarking if codes throw exceptions.
			throw new MathIllegalStateException(LocalizedFormats.SIMPLE_MESSAGE, e.getMessage());
		}

		final double normFactor = 1d / repeatChunk;
		for (int j = 0; j < numMethods; j++) {
			for (int k = 0; k < repeatStat; k++) {
				timesAndResults[j][k][0] *= normFactor;
			}
		}

		return timesAndResults;
	}
		
		/**
	     * Utility class for storing a test label.
	     */
	    public static abstract class RunTest implements Callable<Double> {
	        private final String name;

	        /**
	         * @param name Test name.
	         */
	        public RunTest(String name) {
	            this.name = name;
	        }

	        /**
	         * @return the name of this test.
	         */
	        public String getName() {
	            return name;
	        }

	        /** {@inheritDoc} */
	        public abstract Double call() throws Exception;
	    }

}
