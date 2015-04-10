package org.lemming.tests;

import static org.junit.Assert.assertEquals;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ShortProcessor;

import java.io.FileReader;
import java.util.Properties;

import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.view.Views;

import org.junit.Before;
import org.junit.Test;
import org.lemming.data.Frame;
import org.lemming.data.ImgLib2Frame;
import org.lemming.data.QueueStore;
import org.lemming.inputs.DAXLoader;
import org.lemming.utils.LemMING;

/**
 * Test class to load data from a DAX file format. 
 * NOTE: a test123.DAX files requires a test123.INF file to be located in the same folder.
 * The INF file contains the information for this DAX file, eg. # of frames, frame width & height, etc...
 * 
 * @author Joe Borbely, Thomas Pengo
 *
 */
public class DAXLoaderTest {

	DAXLoader dax;
	QueueStore<ImgLib2Frame<UnsignedShortType>> frames;
	
	@Before
	public void setUp() throws Exception {		
		Properties p = new Properties();
		p.load(new FileReader("test.properties"));
	
		dax = new DAXLoader(p.getProperty("samples.dir")+"daxSample.dax");
		frames = new QueueStore<ImgLib2Frame<UnsignedShortType>>();
		
		dax.setOutput(frames);
	}

	@Test
	public void test() {
		dax.run();
		
		assertEquals(57, frames.getLength());
		
		long[] dimensions = new long[]{dax.width, dax.height, frames.getLength()};
		
		Img<UnsignedShortType> img = ArrayImgs.unsignedShorts(dimensions);
		Cursor<UnsignedShortType> c = img.cursor();
		
		double min=Double.MAX_VALUE, max=-Double.MAX_VALUE;
		while (!frames.isEmpty())
			for (UnsignedShortType t : Views.iterable(frames.get().getPixels())) {
				c.next().set(t);
				min = Math.min(t.get(), min);
				max = Math.max(t.get(), max);
			}
		
		ImagePlus win = ImageJFunctions.show(img);
		win.setTitle("DAX test");
		win.setDisplayRange(min, max);
		
		LemMING.pause(20000);		
	}
}
