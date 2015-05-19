package org.lemming.tests;

import static org.junit.Assert.*;

import java.util.HashMap;
import java.util.Map;

import net.imglib2.util.ValuePair;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.FastMedianFilter;
import org.lemming.modules.IJTiffLoader;
import org.lemming.modules.ImageMath;
import org.lemming.modules.StoreSplitter;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.Pipeline;
import org.lemming.pipeline.Store;

@SuppressWarnings("rawtypes")
public class ImageMathTest {

	private ImageMath im;
	private FastStore frames = new FastStore();
	private FastStore calculated = new FastStore();
	private Pipeline pipe;
	private IJTiffLoader tif;
	private StoreSplitter splitter;
	private FastStore frames1 = new FastStore();
	private FastStore frames2 = new FastStore();
	private FastStore filtered = new FastStore();
	private FastMedianFilter fmf;

	@SuppressWarnings("unchecked")
	@Before
	public void setUp() throws Exception {
		pipe = new Pipeline();
		
		tif = new IJTiffLoader("/home/ronny/Bilder/TubulinAF647.tif");
		tif.setOutput("frames", frames);
		pipe.add(tif);
		
		splitter = new StoreSplitter();
		Map<String,Store> storeMap = new HashMap<>();
		splitter.setInput("frames", frames);
		storeMap.put("frames1", frames1);
		storeMap.put("frames2", frames2);
		splitter.setOutputs(storeMap);
		pipe.add(splitter);	
		
		fmf = new FastMedianFilter(50, true);
		fmf.setInput("frames1", frames1);
		fmf.setOutput("filtered", filtered);
		pipe.add(fmf);
		
		im = new ImageMath(new ValuePair("frames2","filtered"));
		im.setInput("frames2", frames2);
		im.setInput("filtered", filtered);
		im.setOutput("calculated", calculated);
		im.setOperator(ImageMath.operators.SUBSTRACTION);
		pipe.add(im);
	}

	@Test
	public void test() {
		pipe.run();
		//System.out.println(im.getProcessingTime());	
		assertEquals(true,frames.isEmpty());
	}

}
