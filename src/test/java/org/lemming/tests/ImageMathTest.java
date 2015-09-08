package org.lemming.tests;

import static org.junit.Assert.*;

import java.util.HashMap;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.lemming.interfaces.Store;
import org.lemming.modules.ImageLoader;
import org.lemming.modules.ImageMath;
import org.lemming.modules.StoreSplitter;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.Pipeline;
import org.lemming.pipeline.Settings;
import org.lemming.plugins.FastMedianFilter;

import ij.ImagePlus;

@SuppressWarnings("rawtypes")
public class ImageMathTest {

	private ImageMath im;
	private FastStore frames = new FastStore();
	private FastStore calculated = new FastStore();
	private Pipeline pipe;
	private ImageLoader tif;
	private StoreSplitter splitter;
	private FastStore frames1 = new FastStore();
	private FastStore frames2 = new FastStore();
	private FastStore filtered = new FastStore();
	private FastMedianFilter fmf;
	
	@Before
	public void setUp() throws Exception {
		pipe = new Pipeline("test");
		new Settings();
		
		tif = new ImageLoader(new ImagePlus("/Users/ronny/Documents/TubulinAF647.tif"));
		tif.setOutput(frames);
		pipe.add(tif);
		
		splitter = new StoreSplitter();
		Map<Integer,Store> storeMap = new HashMap<>();
		splitter.setInput(frames);
		storeMap.put(frames1.hashCode(), frames1);
		storeMap.put(frames2.hashCode(), frames2);
		splitter.setOutputs(storeMap);
		pipe.add(splitter);	
		
		fmf = new FastMedianFilter(50, true);
		fmf.setInput(frames1);
		fmf.setOutput(filtered);
		pipe.add(fmf);
		
		im = new ImageMath();
		im.setInput(frames2);
		im.setInput(filtered);
		im.setOutput(calculated);
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
