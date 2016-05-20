package org.lemming.modules;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;

import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.NumericType;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.pipeline.SingleRunModule;

/**
 * a class for saving intermediate frames
 * 
 * @author Ronny Sczech
 *
 * @param <T> - data type
 * @param <F> - frame type
 */
public class SaveImages<T extends NumericType<T>, F extends Frame<T>> extends SingleRunModule {
	
	private final String filename;
	private ImageStack stack;

	public SaveImages(String filename){
		this.filename = filename;
	}
	
	@SuppressWarnings("unchecked")
	@Override
	protected void beforeRun(){
		F frame = (F) inputs.get(iterator).peek();
		stack = ImageJFunctions.wrap(frame.getPixels(), "").createEmptyStack();
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public Element processData(Element data) {
		F frame = (F) data;
		if (frame == null) return null;
		stack.addSlice(ImageJFunctions.wrap(frame.getPixels(), "" + frame.getFrameNumber()).getProcessor());
		if (frame.isLast()){ // make the poison pill
			System.out.println("Last Frame saved:" +  frame.getFrameNumber());
			cancel();
		}
		return null;
	}
	
	@Override
	protected void afterRun(){
		ImagePlus out = new ImagePlus();
		out.setOpenAsHyperStack(true);
		out.setStack(stack, 1, 1, stack.getSize());
		IJ.save(out, filename);
	}

	@Override
	public boolean check() {
		return inputs.size()==1;
	}

}
