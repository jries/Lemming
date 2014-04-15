package org.lemming.interfaces;

import java.util.AbstractList;

import org.lemming.data.Frame;
import org.lemming.data.Localization;
import org.lemming.queue.Store;

/**
 * A localizer is transforms Frames into Localizations.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 */
public interface ImageLocalizer<T, F extends Frame<T>> extends Processor<F, AbstractList<Localization>> {
}
