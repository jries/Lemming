package org.lemming.tests;

import static org.junit.Assert.*;

import java.util.HashMap;
import java.util.Map;

import net.imglib2.util.ValuePair;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.IJTiffLoader;
import org.lemming.modules.ImageMath;
import org.lemming.modules.StoreSplitter;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.Pipeline;
import org.lemming.pipeline.Store;

@SuppressWarnings("rawtypes")
public class ImageMathTest {

	private ImageMath im;
	private FastStore frames;
	private FastStore calculated;
	private Pipeline pipe;
	private IJTiffLoader tif;
	private StoreSplitter splitter;
	private FastStore frames1;
	private FastStore frames2;

	@SuppressWarnings("unchecked")
	@Before
	public void setUp() throws Exception {
		pipe = new Pipeline();
		
		frames = new FastStore();
		tif = new IJTiffLoader("/home/ronny/Bilder/TubulinAF647.tif","frames");
		tif.setOutput("frames",frames);
		pipe.add(tif);
		
		splitter = new StoreSplitter("frames");
		Map<String,Store> storeMap = new HashMap<>();
		frames1 = new FastStore();
		frames2 = new FastStore();
		storeMap.put("frames1", frames1);
		storeMap.put("frames2", frames2);
		splitter.setInput("frames", frames);
		splitter.setOutputs(storeMap);
		pipe.add(splitter);		
		
		im = new ImageMath(new ValuePair("frames1","frames2"),"calculated");
		im.setInput("frames1", frames1);
		im.setInput("frames2", frames2);
		calculated = new FastStore();
		im.setOutput("calculated", calculated);
		im.setOperator(ImageMath.operators.ADDITION);
		pipe.add(im);
	}

	@Test
	public void test() {
		pipe.run();
		//System.out.println(im.getProcessingTime());	
		assertEquals(true,frames.isEmpty());
	}

}
