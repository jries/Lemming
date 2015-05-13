package org.lemming.modules;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;

import java.util.Map;

import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.NumericType;

import org.lemming.pipeline.Element;
import org.lemming.pipeline.Frame;
import org.lemming.pipeline.SingleRunModule;
import org.lemming.pipeline.Store;

public class SaveImages<T extends NumericType<T>, F extends Frame<T>> extends SingleRunModule {
	
	private String inputKey;
	private String filename;
	private ImageStack stack;

	public SaveImages(String filename, String inputKey){
		this.filename = filename;
		this.inputKey = inputKey;
	}
	
	@SuppressWarnings({ "rawtypes", "unchecked" })
	@Override
	protected void beforeRun(){
		Store s = inputs.get(inputKey);
		F frame = null;
		while (frame == null)
			frame = (F) s.get();
		stack = ImageJFunctions.wrap(frame.getPixels(), "").createEmptyStack();
		stack.addSlice(ImageJFunctions.wrap(frame.getPixels(), "" + frame.getFrameNumber()).getProcessor());
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public void process(Map<String, Element> data) {
		F frame = (F) data.get(inputKey);
		if (frame == null) return;
		stack.addSlice(ImageJFunctions.wrap(frame.getPixels(), "" + frame.getFrameNumber()).getProcessor());
		if (frame.isLast()){ // make the poison pill
			System.out.println("Last Frame saved:" +  frame.getFrameNumber());
			cancel();
			return;
		}
	}
	
	@Override
	protected void afterRun(){
		ImagePlus out = new ImagePlus();
		out.setOpenAsHyperStack(true);
		out.setStack(stack, 1, 1, stack.getSize());
		IJ.save(out, filename);
	}

}
